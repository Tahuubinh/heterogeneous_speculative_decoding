"""Generate answers with target LM verification and a standalone token n-gram drafter."""

import argparse

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MaxLengthCriteria, StoppingCriteriaList

from evaluation.eval import run_eval, reorg_answer_file
from evaluation.result_utils import (
    build_result_bundle,
    summarize_answer_file,
    write_metrics,
    write_run_config,
)
from model.ngram import TokenNGramModel, greedy_search_ngram


def ngram_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    ngram_lm,
    draft_num_candidate_tokens=10,
    fallback_token_id=100,
):
    input_ids = inputs.input_ids

    output_ids, idx, accept_length_list = model.greedy_search_ngram(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        stopping_criteria=StoppingCriteriaList(
            [MaxLengthCriteria(max_length=len(inputs.input_ids[0]) + max_new_tokens)]
        ),
        draft_num_candidate_tokens=draft_num_candidate_tokens,
        fallback_token_id=fallback_token_id,
        ngram_lm=ngram_lm,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )

    input_len = len(input_ids[0])
    new_token = len(output_ids[0][input_len:])

    # Match other inference scripts: trim post-EOS tail from accounting.
    if tokenizer.eos_token_id in output_ids[0, input_len:].tolist():
        eos_token_index = None
        for i, token_id in enumerate(output_ids[0, input_len:]):
            if token_id == tokenizer.eos_token_id:
                eos_token_index = i
                break

        if eos_token_index is not None:
            invalid_len = len(output_ids[0, input_len:]) - eos_token_index - 1
            if invalid_len > 0:
                accept_length_list[-1] -= invalid_len
                new_token -= invalid_len

    return output_ids, new_token, idx + 1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--ngram-model-path", type=str, required=True)
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
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--draft-num-candidate-tokens",
        type=int,
        default=10,
        help="How many draft tokens n-gram proposes per speculative step.",
    )
    parser.add_argument(
        "--fallback-token-id",
        type=int,
        default=100,
        help="Fallback token id used when n-gram cannot propose any token.",
    )

    args = parser.parse_args()

    if args.draft_num_candidate_tokens < 1:
        raise ValueError("--draft-num-candidate-tokens must be >= 1")

    if args.question_file:
        question_file = args.question_file
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    result_bundle = build_result_bundle(
        model_path=args.model_path,
        drafter_path=args.ngram_model_path,
        draft_tokens=args.draft_num_candidate_tokens,
        question_file=question_file,
        bench_name=args.bench_name,
        version=args.model_id,
        draft_model_label="ngram",
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
            "run_type": "ngram",
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    ngram_lm = TokenNGramModel.load(args.ngram_model_path)
    print(
        "Loaded token n-gram model:",
        f"order={ngram_lm.order}",
        f"tokens={ngram_lm.total_tokens}",
        f"path={args.ngram_model_path}",
    )

    model.greedy_search_ngram = greedy_search_ngram.__get__(model, type(model))

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=ngram_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        ngram_lm=ngram_lm,
        draft_num_candidate_tokens=args.draft_num_candidate_tokens,
        fallback_token_id=args.fallback_token_id,
    )

    reorg_answer_file(answer_file)

    metrics = summarize_answer_file(
        answer_file,
        draft_tokens=args.draft_num_candidate_tokens,
    )
    write_metrics(
        result_bundle["metrics_file"],
        metrics,
        extra={
            "question_file": question_file,
            "answer_file": answer_file,
            "result_dir": result_bundle["result_dir"],
            "run_type": "ngram",
            "ngram_model_path": args.ngram_model_path,
        },
    )
    print(f"Metrics saved to: {result_bundle['metrics_file']}")
