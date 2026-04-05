import argparse
import torch
from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:  # pragma: no cover
    MambaLMHeadModel = None


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


def _sample_or_greedy(logits, do_sample=False, temperature=0.0):
    if do_sample:
        temp = max(float(temperature), 1e-5)
        probs = torch.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _get_logits_from_output(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple) and len(outputs) > 0:
        return outputs[0]
    raise RuntimeError("Unexpected model output format: cannot find logits")


def _draft_with_mamba(prefix_ids, drafter, draft_steps, do_sample=False, temperature=0.0):
    """Generate draft tokens by repeatedly querying Mamba on the running prefix.

    This avoids custom state rollback logic by recomputing from the accepted prefix each SPS step.
    """
    if draft_steps <= 0:
        return prefix_ids[:, :0]

    device = next(drafter.parameters()).device
    draft_seq = prefix_ids.to(device)
    drafted_tokens = []

    for _ in range(draft_steps):
        outputs = drafter(input_ids=draft_seq)
        logits = _get_logits_from_output(outputs)[:, -1, :]
        next_token = _sample_or_greedy(logits, do_sample=do_sample, temperature=temperature)
        drafted_tokens.append(next_token)
        draft_seq = torch.cat((draft_seq, next_token), dim=-1)

    return torch.cat(drafted_tokens, dim=-1)


def _target_verify(prefix_ids, draft_tokens, model, do_sample=False, temperature=0.0):
    """Run one target forward pass over prefix+draft and return selected tokens + matches."""
    cur_len = prefix_ids.shape[1]
    candidate_len = draft_tokens.shape[1]

    candidate_input = torch.cat((prefix_ids, draft_tokens.to(prefix_ids.device)), dim=-1)
    attention_mask = torch.ones_like(candidate_input, device=candidate_input.device)
    outputs = model(input_ids=candidate_input, attention_mask=attention_mask)
    logits = _get_logits_from_output(outputs)

    # candidate_len + 1 target predictions, aligned with HF assisted generation logic.
    relevant_logits = logits[:, cur_len - 1: cur_len + candidate_len, :]
    selected = []
    for i in range(relevant_logits.shape[1]):
        next_tok = _sample_or_greedy(
            relevant_logits[:, i, :],
            do_sample=do_sample,
            temperature=temperature,
        )
        selected.append(next_tok)
    selected_tokens = torch.cat(selected, dim=-1)

    if candidate_len == 0:
        return selected_tokens[:, :1], 0

    candidate_new_tokens = draft_tokens.to(selected_tokens.device)
    mismatch_prefix = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1)
    n_matches = int(mismatch_prefix.sum().item())
    valid_tokens = selected_tokens[:, : n_matches + 1]
    return valid_tokens, n_matches

def sps_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    do_sample=False,
    temperature=0.0,
    drafter=None,
    draft_tokens=None,
):
    """
    Forward pass for Speculative Parallel Sampling (SPS).
    """
    if drafter is None:
        raise ValueError("drafter (Mamba model) is required")

    prefix_ids = inputs.input_ids
    eos_token_id = tokenizer.eos_token_id

    steps = 0
    new_token = 0
    accept_length_list = []
    output_ids = prefix_ids

    adaptive = draft_tokens is None
    current_draft_tokens = 5 if adaptive else int(draft_tokens)

    while new_token < max_new_tokens:
        steps += 1
        remaining = max_new_tokens - new_token
        max_matches = max(remaining - 1, 0)
        effective_draft_steps = min(current_draft_tokens, max_matches)

        draft_ids = _draft_with_mamba(
            output_ids,
            drafter,
            effective_draft_steps,
            do_sample=do_sample,
            temperature=temperature,
        )

        valid_tokens, n_matches = _target_verify(
            output_ids,
            draft_ids,
            model,
            do_sample=do_sample,
            temperature=temperature,
        )

        output_ids = torch.cat((output_ids, valid_tokens.to(output_ids.device)), dim=-1)
        accepted = int(valid_tokens.shape[1])
        new_token += accepted
        accept_length_list.append(accepted)

        if adaptive:
            if n_matches == current_draft_tokens:
                current_draft_tokens += 2
            else:
                current_draft_tokens = max(1, current_draft_tokens - 1)

        if eos_token_id is not None and output_ids[0, -1].item() == eos_token_id:
            break

    return output_ids, new_token, steps, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--drafter-path", type=str, required=True)
    parser.add_argument(
        "--drafter-tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer for draft Mamba model. Defaults to target model tokenizer.",
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
    print(f"Output saved to: {answer_file}")

    # Load the Target Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    if MambaLMHeadModel is None:
        raise ImportError(
            "mamba_ssm is not available. Please install mamba-ssm before running inference_sps_mamba.py"
        )

    # Load the Draft Model (Mamba)
    drafter = MambaLMHeadModel.from_pretrained(args.drafter_path)
    drafter = drafter.to(device="cuda", dtype=str_to_torch_dtype(args.dtype))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    draft_tokenizer_path = args.drafter_tokenizer_path or args.model_path
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_tokenizer_path)

    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError(
            "Target tokenizer and draft tokenizer vocabularies differ. "
            "Please provide a tokenizer compatible with both models via --drafter-tokenizer-path."
        )
    
    # --- CRITICAL FIXES FOR PYTHIA/LLAMA ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure target model uses a valid padding token for batched forward.
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(drafter, "config"):
        drafter.config.pad_token_id = tokenizer.pad_token_id

    # Configure how many draft tokens the assistant proposes each speculative step.
    # None => adaptive heuristic (same behavior as the original file);
    # positive integer => fixed constant per speculative step.
    if args.draft_tokens is None:
        print("Draft config -> adaptive heuristic, initial draft_tokens=5")
    else:
        print(f"Draft config -> fixed draft_tokens={args.draft_tokens}")

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
        draft_tokens=args.draft_tokens,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)