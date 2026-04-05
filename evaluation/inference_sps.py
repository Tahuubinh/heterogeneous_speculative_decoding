import argparse
import torch
from evaluation.eval import run_eval, reorg_answer_file
from evaluation.result_utils import (
    build_result_bundle,
    summarize_answer_file,
    write_metrics,
    write_run_config,
)
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding


def parse_optional_int(value):
    if value is None:
        return None
    value_str = str(value).strip().lower()
    if value_str in {"none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("draft-tokens must be a positive integer or None") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("draft-tokens must be > 0 or None")
    return parsed

def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    """
    Forward pass for Speculative Parallel Sampling (SPS).
    """
    input_ids = inputs.input_ids
    
    # Generate tokens using assisted decoding
    # Note: Use **inputs to pass attention_mask correctly
    output_ids, idx, accept_length_list = model.generate(
        **inputs, 
        assistant_model=drafter, 
        do_sample=do_sample, 
        temperature=temperature if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    
    new_token = len(output_ids[0][len(input_ids[0]):])
    # Returns exactly what eval.py expects
    return output_ids, new_token, idx + 1, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--drafter-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--draft-tokens",
        type=parse_optional_int,
        default=None,
        help=(
            "Draft tokens per speculative step. Set to None (default) for adaptive heuristic "
            "behavior like inference_sps_old.py, or set a positive integer for fixed constant drafting."
        ),
    )
    
    args = parser.parse_args()

    # Apply the SPS decoding patch
    GenerationMixin.assisted_decoding = assisted_decoding

    if args.question_file:
        question_file = args.question_file
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    result_bundle = build_result_bundle(
        model_path=args.model_path,
        drafter_path=args.drafter_path,
        draft_tokens=args.draft_tokens,
        question_file=question_file,
        bench_name=args.bench_name,
        version=args.model_id,
        answer_file_override=args.answer_file,
    )
    answer_file = result_bundle["answer_file"]

    print(f"Reading from: {question_file}")
    print(f"Output saved to: {answer_file}")
    print(f"Result directory: {result_bundle['result_dir']}")

    write_run_config(
        result_bundle["config_file"],
        vars(args),
        extra={
            "question_file": question_file,
            "result_dir": result_bundle["result_dir"],
        },
    )

    # Load the Target Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # Load the Draft Model
    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # --- CRITICAL FIXES FOR PYTHIA/LLAMA ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure both models agree on the pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    drafter.config.pad_token_id = tokenizer.pad_token_id

    # Configure how many draft tokens the assistant proposes each speculative step.
    # None => adaptive heuristic (same behavior as the original file);
    # positive integer => fixed constant per speculative step.
    if args.draft_tokens is None:
        drafter.generation_config.num_assistant_tokens_schedule = "heuristic"
        print(
            "Draft config -> adaptive (heuristic), initial num_assistant_tokens="
            f"{drafter.generation_config.num_assistant_tokens}"
        )
    else:
        drafter.generation_config.num_assistant_tokens = args.draft_tokens
        drafter.generation_config.num_assistant_tokens_schedule = "constant"
        print(
            "Draft config -> fixed, num_assistant_tokens="
            f"{drafter.generation_config.num_assistant_tokens}, schedule=constant"
        )

    model.eval()
    drafter.eval()

    do_sample = True if args.temperature > 0 else False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=sps_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=drafter,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)

    metrics = summarize_answer_file(answer_file, draft_tokens=args.draft_tokens)
    write_metrics(
        result_bundle["metrics_file"],
        metrics,
        extra={
            "question_file": question_file,
            "answer_file": answer_file,
            "result_dir": result_bundle["result_dir"],
        },
    )
    print(f"Metrics saved to: {result_bundle['metrics_file']}")