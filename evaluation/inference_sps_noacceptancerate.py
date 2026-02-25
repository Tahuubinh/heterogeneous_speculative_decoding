import argparse
import torch
import time
from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    """
    English Comment:
    A manual speculative decoding loop compatible with Transformers 4.56.1.
    This replaces the broken decoding.py logic while preserving the return 
    interface expected by eval.py.
    """
    input_ids = inputs.input_ids
    cur_len = input_ids.shape[-1]
    max_length = cur_len + max_new_tokens
    
    # Using modern DynamicCache objects for Gemma 2 compatibility
    past_key_values = DynamicCache()
    drafter_past_key_values = DynamicCache()
    
    # Setting the speculative window (Gamma)
    gamma = 5 
    accept_length_list = []
    steps = 0

    while input_ids.shape[-1] < max_length:
        steps += 1
        
        # 1. Draft Generation (Predict K tokens)
        with torch.no_grad():
            draft_output = drafter.generate(
                input_ids=input_ids,
                max_new_tokens=gamma,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                use_cache=True,
                past_key_values=drafter_past_key_values,
                return_dict_in_generate=True,
            )
        
        candidate_ids = draft_output.sequences[:, cur_len:]
        actual_gamma = candidate_ids.shape[-1]
        
        # 2. Target Verification (Parallel forward pass)
        with torch.no_grad():
            full_input_ids = torch.cat([input_ids, candidate_ids], dim=-1)
            outputs = model(
                full_input_ids, 
                use_cache=True, 
                past_key_values=past_key_values
            )
            # Compare target choices with draft tokens
            target_selected_ids = outputs.logits[:, cur_len - 1 : -1, :].argmax(dim=-1)

        # 3. Match Counting (Acceptance Logic)
        matches = (candidate_ids == target_selected_ids).all(dim=0)
        n_matches = 0
        for m in matches:
            if m: n_matches += 1
            else: break
            
        # 4. Metrics & Cache Rollback
        accept_length_list.append(n_matches + 1)
        new_len = cur_len + n_matches + 1
        input_ids = full_input_ids[:, :new_len]
        
        # Crop the modern Cache objects
        past_key_values.crop(new_len)
        drafter_past_key_values.crop(new_len)
        cur_len = new_len
        
        if (input_ids == tokenizer.eos_token_id).any():
            break

    # Return exactly what eval.py's get_model_answers() expects
    new_token_count = input_ids.shape[-1] - inputs.input_ids.shape[-1]
    return input_ids, new_token_count, steps, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (Keep all your existing arguments: model-path, drafter-path, model-id, etc.)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--drafter-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    # Add other necessary arguments for run_eval compatibility
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    
    args = parser.parse_args()

    # Determine input/output paths
    question_file = args.question_file or f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        suffix = args.question_file.split('/')[-1].replace('.jsonl', '') if args.question_file else args.bench_name
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}_{suffix}.jsonl"

    # Load Target and Draft Models
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=str_to_torch_dtype(args.dtype), device_map="auto")
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter_path, torch_dtype=str_to_torch_dtype(args.dtype), device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Call the ORIGINAL run_eval
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
        drafter=drafter, # Passed as kwargs to sps_forward
        temperature=args.temperature,
    )

    reorg_answer_file(answer_file)