import argparse
from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding

def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    """
    Forward pass for Speculative Parallel Sampling (SPS).
    Verification is handled by the target model using draft tokens from the assistant model.
    """
    input_ids = inputs.input_ids
    model.generation_config.max_new_tokens = max_new_tokens
    
    # Generate tokens using assisted decoding (Speculative Decoding)
    output_ids, idx, accept_length_list = model.generate(
        **inputs, 
        generation_config=model.generation_config, 
        assistant_model=drafter, 
        do_sample=do_sample, 
        temperature=temperature
    )
    new_token = len(output_ids[0][len(input_ids[0]):])
    return output_ids, new_token, idx+1, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the target model (e.g., Gemma-9B)")
    parser.add_argument("--drafter-path", type=str, required=True, help="Path to the draft model (e.g., Gemma-2B)")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-file", type=str, help="Path to the custom question.jsonl file")
    parser.add_argument("--question-begin", type=int, help="Starting index for questions")
    parser.add_argument("--question-end", type=int, help="Ending index for questions")
    parser.add_argument("--answer-file", type=str, help="Path to save the generated answers")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float64", "float16", "bfloat16"])
    
    args = parser.parse_args()

    # Apply the SPS decoding patch to the Transformers library
    GenerationMixin.assisted_decoding = assisted_decoding

    # Handle input file path logic
    if args.question_file:
        question_file = args.question_file
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    # Handle output file path naming logic
    if args.answer_file:
        answer_file = args.answer_file
    else:
        suffix = args.question_file.split('/')[-1].replace('.jsonl', '') if args.question_file else args.bench_name
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}_{suffix}.jsonl"

    print(f"Reading from: {question_file}")
    print(f"Output saved to: {answer_file}")

    # Load the Target Model (e.g., 9B) onto the GPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # Load the Draft Model (e.g., 2B) onto the GPU
    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    drafter.eval()

    do_sample = True if args.temperature > 0 else False

    # Execute evaluation loop
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