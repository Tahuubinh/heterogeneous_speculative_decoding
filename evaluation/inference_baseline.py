"""Generate answers with local models.

Usage:
python3 evaluation/inference_baseline.py --model-path EleutherAI/pythia-6.9b --model-id pythia-6.9b --question-file data/spec_bench/split_categories/summarization.jsonl
"""
import argparse
import torch
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file

from transformers import AutoModelForCausalLM, AutoTokenizer


def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    # inputs contains both input_ids and attention_mask
    input_ids = inputs.input_ids
    
    # Use **inputs to pass attention_mask to avoid "unexpected behavior" warnings
    # Set temperature to None if do_sample is False to avoid UserWarnings
    output_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    new_token = len(output_ids[0][len(input_ids[0]):])
    step = new_token
    accept_length_list = [1] * new_token
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
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
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
    )

    args = parser.parse_args()

    if args.question_file:
        question_file = args.question_file
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        suffix = args.question_file.split('/')[-1].replace('.jsonl', '') if args.question_file else args.bench_name
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}_{suffix}.jsonl"

    print(f"Reading from: {question_file}")
    print(f"Output to: {answer_file}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Fix for models that don't have a default pad_token (like Pythia/Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure model knows its pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)