import argparse
import torch
import time
from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    input_ids = inputs.input_ids
    device = input_ids.device
    # Initial sequence length
    prefix_len = input_ids.shape[-1]
    max_length = prefix_len + max_new_tokens
    
    # Initialize modern DynamicCache objects
    past_key_values = DynamicCache()
    drafter_past_key_values = DynamicCache()
    
    # Speculative window size (K)
    gamma = 5 
    accept_length_list = []
    steps = 0

    # Warmup: Populate KV caches with all prompt tokens EXCEPT the last one.
    # Both models stop at prefix_len-1 so that each main-loop iteration can
    # re-process the last accepted token and get a clean, unconditional logit
    # for the first draft-token position.
    with torch.no_grad():
        if prefix_len > 1:
            prefill_cache_pos = torch.arange(prefix_len - 1, device=device)
            model(input_ids[:, :-1], past_key_values=past_key_values, cache_position=prefill_cache_pos, use_cache=True)
            drafter(input_ids[:, :-1], past_key_values=drafter_past_key_values, cache_position=prefill_cache_pos, use_cache=True)

    while input_ids.shape[-1] < max_length:
        steps += 1
        current_num_tokens = input_ids.shape[-1]
        
        # 1. Draft Generation: Predict K candidate tokens via manual forward passes
        with torch.no_grad():
            draft_token_ids = []
            # The last accepted token is the starting point for drafting
            last_token = input_ids[:, -1:]
            for _ in range(gamma):
                draft_pos = torch.tensor(
                    [drafter_past_key_values.get_seq_length()], device=device
                )
                draft_out = drafter(
                    last_token,
                    past_key_values=drafter_past_key_values,
                    cache_position=draft_pos,
                    use_cache=True,
                )
                next_token = draft_out.logits[:, -1:, :].argmax(dim=-1)
                draft_token_ids.append(next_token)
                last_token = next_token
                if (next_token == tokenizer.eos_token_id).any():
                    break

        candidate_ids = torch.cat(draft_token_ids, dim=-1)  # [1, actual_gamma]
        actual_gamma = candidate_ids.shape[-1]
        
        if actual_gamma == 0:
            break

        # 2. Target Verification: Parallel validation using the large model.
        # We prepend the last accepted token so that logits[0] is the target's
        # UNCONDITIONAL prediction for the first draft-token position, logits[1]
        # is conditioned only on d_1 being accepted, etc.
        with torch.no_grad():
            last_accepted = input_ids[:, current_num_tokens - 1 : current_num_tokens]  # [1, 1]
            verify_input = torch.cat([last_accepted, candidate_ids], dim=-1)  # [1, 1 + actual_gamma]
            # Positions: last accepted sits at current_num_tokens-1; drafts follow
            cache_position = torch.arange(
                current_num_tokens - 1, current_num_tokens + actual_gamma, device=device
            )
            outputs = model(
                verify_input,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            # Shape [1, 1+actual_gamma]. logits[:, i] predicts position current_num_tokens+i.
            target_selected_ids = outputs.logits.argmax(dim=-1)

        # 3. Acceptance Logic: Compare draft tokens vs target choices.
        # target_selected_ids[0, :-1] has shape [actual_gamma], same as candidate_ids[0].
        # target_selected_ids[0, i] = what target predicts at current_num_tokens+i
        #   conditioned on d_1..d_i already being accepted (correct for greedy SPS).
        matches = (candidate_ids[0] == target_selected_ids[0, :-1]).cpu()
        
        n_matches = 0
        for m in matches:
            if m: n_matches += 1
            else: break
            
        # 4. Update Sequence and Synchronize Caches
        accept_length_list.append(n_matches + 1)
        
        # The next token is the target's first mismatch or its final prediction
        next_token_id = target_selected_ids[:, n_matches : n_matches + 1]
        
        # Update the main sequence
        input_ids = torch.cat([input_ids[:, :current_num_tokens + n_matches], next_token_id], dim=-1)
        
        # Roll back caches so both sit at new_len-1 entries (positions 0..new_len-2).
        # In the next iteration, both models will re-process input_ids[:, -1:]
        # (the newly accepted token) as the first step.
        new_len = input_ids.shape[-1]
        past_key_values.crop(new_len - 1)
        drafter_past_key_values.crop(new_len - 1)
        
        if (next_token_id == tokenizer.eos_token_id).any():
            break

    total_generated = input_ids.shape[-1] - prefix_len
    return input_ids, total_generated, steps, accept_length_list

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