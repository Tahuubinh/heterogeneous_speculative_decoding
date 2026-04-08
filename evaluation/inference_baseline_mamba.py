"""Generate answers with a standalone Mamba model (baseline for draft-only throughput).

Usage:
python3 evaluation/inference_baseline_mamba.py \
  --model-path state-spaces/mamba-130m \
  --tokenizer-path EleutherAI/gpt-neox-20b \
  --model-id mamba-130m-baseline \
  --question-file data/spec_bench/split_categories/summarization.jsonl
"""

import argparse

import torch
from fastchat.utils import str_to_torch_dtype
from transformers import AutoTokenizer

from evaluation.eval import reorg_answer_file, run_eval
from evaluation.result_utils import (
    build_result_bundle,
    summarize_answer_file,
    write_metrics,
    write_run_config,
)

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:  # pragma: no cover
    MambaLMHeadModel = None


def baseline_mamba_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    temperature=0.0,
    do_sample=False,
    use_cuda_graphs=False,
):
    input_ids = inputs.input_ids
    max_length = int(input_ids.shape[1] + max_new_tokens)

    gen_kwargs = {
        "input_ids": input_ids,
        "max_length": max_length,
        "cg": bool(use_cuda_graphs),
    }

    # Mamba's generation supports sampling args; keep greedy by default for fair throughput runs.
    if do_sample:
        gen_kwargs.update({
            "top_k": 50,
            "temperature": max(float(temperature), 1e-5),
        })

    outputs = model.generate(**gen_kwargs)
    output_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs

    new_token = len(output_ids[0][len(input_ids[0]):])
    step = new_token
    accept_length_list = [1] * new_token
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        help="Tokenizer compatible with the Mamba checkpoint.",
    )
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
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        help="Enable CUDA Graph decoding path in mamba_ssm generate().",
    )

    args = parser.parse_args()

    if args.question_file:
        question_file = args.question_file
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    result_bundle = build_result_bundle(
        model_path=args.model_path,
        drafter_path="baseline_mamba",
        draft_tokens=None,
        question_file=question_file,
        bench_name=args.bench_name,
        version=args.model_id,
        draft_model_label="baseline_mamba",
        draft_bucket_label="none",
        answer_file_override=args.answer_file,
    )
    answer_file = result_bundle["answer_file"]

    print(f"Reading from: {question_file}")
    print(f"Output to: {answer_file}")
    print(f"Result directory: {result_bundle['result_dir']}")

    write_run_config(
        result_bundle["config_file"],
        vars(args),
        extra={
            "question_file": question_file,
            "result_dir": result_bundle["result_dir"],
            "run_type": "baseline_mamba",
        },
    )

    if MambaLMHeadModel is None:
        raise ImportError(
            "mamba_ssm is not available. Please install mamba-ssm before running inference_baseline_mamba.py"
        )

    model = MambaLMHeadModel.from_pretrained(args.model_path)
    model = model.to(device="cuda", dtype=str_to_torch_dtype(args.dtype))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    do_sample = args.temperature > 0

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_mamba_forward,
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
        use_cuda_graphs=args.use_cuda_graphs,
    )

    reorg_answer_file(answer_file)

    metrics = summarize_answer_file(answer_file, draft_tokens=None)
    write_metrics(
        result_bundle["metrics_file"],
        metrics,
        extra={
            "question_file": question_file,
            "answer_file": answer_file,
            "result_dir": result_bundle["result_dir"],
            "run_type": "baseline_mamba",
        },
    )
    print(f"Metrics saved to: {result_bundle['metrics_file']}")
