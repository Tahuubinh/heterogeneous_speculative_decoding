import argparse
import copy
import torch
from evaluation.eval import run_eval, reorg_answer_file
from evaluation.result_utils import (
    build_result_bundle,
    summarize_answer_file,
    write_metrics,
    write_run_config,
)
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import _crop_past_key_values

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:  # pragma: no cover
    MambaLMHeadModel = None

try:
    from mamba_ssm.utils.generation import InferenceParams
except Exception:  # pragma: no cover
    InferenceParams = None

try:
    from mamba_ssm.utils.generation import update_graph_cache as mamba_update_graph_cache
except Exception:  # pragma: no cover
    mamba_update_graph_cache = None


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


def _mamba_forward(drafter, input_ids, inference_params=None):
    if inference_params is None:
        return drafter(input_ids=input_ids)
    return drafter(input_ids=input_ids, inference_params=inference_params)


def _last_token_logits(outputs):
    logits = _get_logits_from_output(outputs)
    if logits.ndim == 3:
        return logits[:, -1, :]
    return logits


def _clone_mamba_inference_params(inference_params, backup_mode="tensor_clone"):
    if inference_params is None:
        return None
    if backup_mode == "deepcopy":
        return copy.deepcopy(inference_params)

    if InferenceParams is None:
        raise RuntimeError("InferenceParams is required for tensor_clone backup mode")

    key_value_memory_dict = {}
    for layer_idx, state_value in inference_params.key_value_memory_dict.items():
        if isinstance(state_value, tuple):
            key_value_memory_dict[layer_idx] = tuple(
                tensor.clone() if torch.is_tensor(tensor) else copy.deepcopy(tensor)
                for tensor in state_value
            )
        elif torch.is_tensor(state_value):
            key_value_memory_dict[layer_idx] = state_value.clone()
        else:
            key_value_memory_dict[layer_idx] = copy.deepcopy(state_value)

    lengths_per_sample = None
    if inference_params.lengths_per_sample is not None:
        lengths_per_sample = inference_params.lengths_per_sample.clone()

    return InferenceParams(
        max_seqlen=inference_params.max_seqlen,
        max_batch_size=inference_params.max_batch_size,
        seqlen_offset=inference_params.seqlen_offset,
        batch_size_offset=inference_params.batch_size_offset,
        key_value_memory_dict=key_value_memory_dict,
        lengths_per_sample=lengths_per_sample,
    )


def _build_mamba_state_from_prefix(
    drafter,
    prefix_ids,
    max_new_tokens,
    state_seqlen_buffer,
    use_mamba_cuda_graphs=False,
    mamba_cg_warmups=2,
):
    if InferenceParams is None:
        raise RuntimeError(
            "InferenceParams is unavailable. Install a mamba_ssm version exposing mamba_ssm.utils.generation.InferenceParams"
        )

    batch_size = int(prefix_ids.shape[0])
    max_seqlen = int(prefix_ids.shape[1] + max_new_tokens + state_seqlen_buffer)

    if use_mamba_cuda_graphs:
        if mamba_update_graph_cache is None:
            raise RuntimeError(
                "CUDA Graphs requested but update_graph_cache is unavailable in current mamba_ssm version"
            )

        cg_cache = getattr(drafter, "_spec_cg_cache", None)
        param = next(drafter.parameters())
        cg_cache = mamba_update_graph_cache(
            drafter,
            cg_cache,
            batch_size=batch_size,
            seqlen_og=int(prefix_ids.shape[1]),
            max_seqlen=max_seqlen,
            decoding_seqlens=(1,),
            dtype=param.dtype,
            n_warmups=max(0, int(mamba_cg_warmups)),
        )
        setattr(drafter, "_spec_cg_cache", cg_cache)
        inference_params = cg_cache.inference_params
        outputs = drafter(input_ids=prefix_ids, inference_params=inference_params, num_last_tokens=1)
        next_logits = _last_token_logits(outputs)
        inference_params.seqlen_offset += int(prefix_ids.shape[1])
        if inference_params.lengths_per_sample is not None:
            inference_params.lengths_per_sample[:] = inference_params.seqlen_offset
        return {
            "inference_params": inference_params,
            "next_logits": next_logits,
            "cg_cache": cg_cache,
        }

    inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
    outputs = _mamba_forward(drafter, prefix_ids, inference_params=inference_params)
    next_logits = _last_token_logits(outputs)
    return {
        "inference_params": inference_params,
        "next_logits": next_logits,
        "cg_cache": None,
    }


def _mamba_step_with_state(drafter, state, token_ids):
    if state.get("cg_cache") is not None:
        batch_size = int(token_ids.shape[0])
        seqlen_offset = int(state["inference_params"].seqlen_offset)
        position_ids = torch.full(
            (batch_size, token_ids.shape[1]),
            seqlen_offset,
            dtype=torch.long,
            device=token_ids.device,
        )
        logits = state["cg_cache"].run(token_ids, position_ids, seqlen_offset)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        state["inference_params"].seqlen_offset += int(token_ids.shape[1])
        if state["inference_params"].lengths_per_sample is not None:
            state["inference_params"].lengths_per_sample[:] = state["inference_params"].seqlen_offset
        state["next_logits"] = logits
        return state

    outputs = _mamba_forward(drafter, token_ids, inference_params=state["inference_params"])
    state["next_logits"] = _last_token_logits(outputs)
    return state


def _consume_tokens_with_state(drafter, state, tokens):
    if tokens.shape[1] == 0:
        return state

    if state.get("cg_cache") is not None and tokens.shape[1] > 1:
        for i in range(tokens.shape[1]):
            state = _mamba_step_with_state(drafter, state, tokens[:, i:i + 1])
        return state

    state = _mamba_step_with_state(drafter, state, tokens)
    return state


def _draft_with_mamba_state(
    drafter,
    state,
    draft_steps,
    do_sample=False,
    temperature=0.0,
):
    if draft_steps <= 0:
        batch_size = int(state["next_logits"].shape[0])
        return torch.empty((batch_size, 0), dtype=torch.long, device=state["next_logits"].device)

    drafted_tokens = []
    cur_logits = state["next_logits"]
    for _ in range(draft_steps):
        next_token = _sample_or_greedy(cur_logits, do_sample=do_sample, temperature=temperature)
        drafted_tokens.append(next_token)
        state = _mamba_step_with_state(drafter, state, next_token)
        cur_logits = state["next_logits"]

    state["next_logits"] = cur_logits
    return torch.cat(drafted_tokens, dim=-1)


def _target_verify(
    prefix_ids,
    draft_tokens,
    model,
    past_key_values=None,
    use_target_kv_cache=True,
    do_sample=False,
    temperature=0.0,
):
    """Run one target forward pass over prefix+draft and return selected tokens + matches."""
    cur_len = prefix_ids.shape[1]
    candidate_len = draft_tokens.shape[1]

    if use_target_kv_cache and past_key_values is not None:
        candidate_input = torch.cat((prefix_ids[:, -1:], draft_tokens.to(prefix_ids.device)), dim=-1)
        outputs = model(
            input_ids=candidate_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        relevant_logits = _get_logits_from_output(outputs)
    else:
        candidate_input = torch.cat((prefix_ids, draft_tokens.to(prefix_ids.device)), dim=-1)
        outputs = model(
            input_ids=candidate_input,
            use_cache=use_target_kv_cache,
        )
        logits = _get_logits_from_output(outputs)
        # candidate_len + 1 target predictions, aligned with HF assisted generation logic.
        relevant_logits = logits[:, cur_len - 1: cur_len + candidate_len, :]

    new_past_key_values = outputs.past_key_values if use_target_kv_cache else None
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
        return selected_tokens[:, :1], 0, new_past_key_values

    candidate_new_tokens = draft_tokens.to(selected_tokens.device)
    mismatch_prefix = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1)
    n_matches = int(mismatch_prefix.sum().item())
    valid_tokens = selected_tokens[:, : n_matches + 1]
    return valid_tokens, n_matches, new_past_key_values

def sps_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    do_sample=False,
    temperature=0.0,
    drafter=None,
    draft_tokens=None,
    mamba_state_strategy="recompute",
    mamba_state_fallback_reset_replay=True,
    mamba_state_seqlen_buffer=32,
    mamba_state_backup_mode="tensor_clone",
    use_target_kv_cache=True,
    use_mamba_cuda_graphs=False,
    mamba_cg_warmups=2,
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
    mamba_state = None
    target_past_key_values = None

    adaptive = draft_tokens is None
    current_draft_tokens = 5 if adaptive else int(draft_tokens)
    mamba_cg_enabled = bool(use_mamba_cuda_graphs and mamba_state_strategy != "stateful_restore")

    while new_token < max_new_tokens:
        steps += 1
        remaining = max_new_tokens - new_token
        max_matches = max(remaining - 1, 0)
        effective_draft_steps = min(current_draft_tokens, max_matches)

        if mamba_state_strategy == "recompute":
            # Recompute strategy: rebuild Mamba state from accepted prefix once per SPS step,
            # then draft incrementally from cached state.
            temp_mamba_state = _build_mamba_state_from_prefix(
                drafter,
                output_ids.to(next(drafter.parameters()).device),
                max_new_tokens,
                mamba_state_seqlen_buffer,
                use_mamba_cuda_graphs=mamba_cg_enabled,
                mamba_cg_warmups=mamba_cg_warmups,
            )
            draft_ids = _draft_with_mamba_state(
                drafter,
                temp_mamba_state,
                effective_draft_steps,
                do_sample=do_sample,
                temperature=temperature,
            )
            pre_draft_state_backup = None
        else:
            if mamba_state is None:
                mamba_state = _build_mamba_state_from_prefix(
                    drafter,
                    output_ids.to(next(drafter.parameters()).device),
                    max_new_tokens,
                    mamba_state_seqlen_buffer,
                    use_mamba_cuda_graphs=mamba_cg_enabled,
                    mamba_cg_warmups=mamba_cg_warmups,
                )

            pre_draft_state_backup = None
            if mamba_state_strategy == "stateful_restore":
                try:
                    pre_draft_state_backup = {
                        "inference_params": _clone_mamba_inference_params(
                            mamba_state["inference_params"],
                            backup_mode=mamba_state_backup_mode,
                        ),
                        "next_logits": mamba_state["next_logits"].clone(),
                    }
                except Exception:
                    if not mamba_state_fallback_reset_replay:
                        raise
                    pre_draft_state_backup = None

            draft_ids = _draft_with_mamba_state(
                drafter,
                mamba_state,
                effective_draft_steps,
                do_sample=do_sample,
                temperature=temperature,
            )

        valid_tokens, n_matches, target_past_key_values = _target_verify(
            output_ids,
            draft_ids,
            model,
            past_key_values=target_past_key_values,
            use_target_kv_cache=use_target_kv_cache,
            do_sample=do_sample,
            temperature=temperature,
        )

        if use_target_kv_cache and target_past_key_values is not None:
            keep_cache_length = int(output_ids.shape[1] + n_matches)
            target_past_key_values = _crop_past_key_values(model, target_past_key_values, keep_cache_length)

        output_ids = torch.cat((output_ids, valid_tokens.to(output_ids.device)), dim=-1)
        accepted = int(valid_tokens.shape[1])
        new_token += accepted
        accept_length_list.append(accepted)

        if mamba_state_strategy in {"reset_replay", "stateful_restore"}:
            drafted_count = int(draft_ids.shape[1]) if draft_ids is not None else 0
            full_accept = n_matches == drafted_count
            valid_tokens_on_drafter = valid_tokens.to(next(drafter.parameters()).device)

            if full_accept:
                extra_target_token = valid_tokens_on_drafter[:, -1:]
                mamba_state = _consume_tokens_with_state(drafter, mamba_state, extra_target_token)
            else:
                if mamba_state_strategy == "reset_replay":
                    mamba_state = _build_mamba_state_from_prefix(
                        drafter,
                        output_ids.to(next(drafter.parameters()).device),
                        max_new_tokens,
                        mamba_state_seqlen_buffer,
                        use_mamba_cuda_graphs=mamba_cg_enabled,
                        mamba_cg_warmups=mamba_cg_warmups,
                    )
                elif mamba_state_strategy == "stateful_restore" and pre_draft_state_backup is not None:
                    mamba_state["inference_params"] = pre_draft_state_backup["inference_params"]
                    mamba_state["next_logits"] = pre_draft_state_backup["next_logits"]
                    mamba_state = _consume_tokens_with_state(drafter, mamba_state, valid_tokens_on_drafter)
                elif mamba_state_fallback_reset_replay:
                    mamba_state = _build_mamba_state_from_prefix(
                        drafter,
                        output_ids.to(next(drafter.parameters()).device),
                        max_new_tokens,
                        mamba_state_seqlen_buffer,
                        use_mamba_cuda_graphs=mamba_cg_enabled,
                        mamba_cg_warmups=mamba_cg_warmups,
                    )
                else:
                    raise RuntimeError(
                        "State rollback required but unavailable. Enable --mamba-state-fallback-reset-replay."
                    )

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
    parser.add_argument(
        "--mamba-state-strategy",
        type=str,
        default="recompute",
        choices=["recompute", "reset_replay", "stateful_restore"],
        help=(
            "Mamba draft-state handling strategy: recompute (safe baseline), "
            "reset_replay (reset and replay accepted prefix on reject), "
            "stateful_restore (try backup/restore state, fallback optional)."
        ),
    )
    parser.add_argument(
        "--mamba-state-fallback-reset-replay",
        action="store_true",
        help="Only for stateful_restore: if deepcopy/restore fails, fallback to reset+replay.",
    )
    parser.add_argument(
        "--mamba-state-seqlen-buffer",
        type=int,
        default=32,
        help="Extra headroom for Mamba InferenceParams.max_seqlen in stateful modes.",
    )
    parser.add_argument(
        "--mamba-state-backup-mode",
        type=str,
        default="tensor_clone",
        choices=["tensor_clone", "deepcopy"],
        help=(
            "Backup method for stateful_restore. tensor_clone is usually faster than deepcopy. "
            "deepcopy is slower but can be used for debugging correctness."
        ),
    )
    parser.add_argument(
        "--disable-target-kv-cache",
        action="store_true",
        help="Disable target-model KV cache in verification (debug only; much slower).",
    )
    parser.add_argument(
        "--mamba-use-cuda-graphs",
        action="store_true",
        help="Enable CUDA Graphs for Mamba draft decode path (experimental, can improve draft throughput).",
    )
    parser.add_argument(
        "--mamba-cg-warmups",
        type=int,
        default=2,
        help="Number of warmup iterations used when capturing Mamba CUDA Graphs.",
    )
    
    args = parser.parse_args()

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
    print(
        "Mamba state config -> strategy="
        f"{args.mamba_state_strategy}, "
        "fallback_reset_replay="
        f"{args.mamba_state_fallback_reset_replay}, "
        "seqlen_buffer="
        f"{args.mamba_state_seqlen_buffer}, "
        "backup_mode="
        f"{args.mamba_state_backup_mode}, "
        "target_kv_cache="
        f"{not args.disable_target_kv_cache}, "
        "mamba_cuda_graphs="
        f"{args.mamba_use_cuda_graphs}, "
        "mamba_cg_warmups="
        f"{args.mamba_cg_warmups}"
    )

    model.eval()
    drafter.eval()

    do_sample = True if args.temperature > 0 else False

    if args.mamba_use_cuda_graphs and args.mamba_state_strategy == "stateful_restore":
        print("Warning: disabling Mamba CUDA Graphs for stateful_restore to keep state snapshot/restore semantics stable.")

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
        mamba_state_strategy=args.mamba_state_strategy,
        mamba_state_fallback_reset_replay=args.mamba_state_fallback_reset_replay,
        mamba_state_seqlen_buffer=args.mamba_state_seqlen_buffer,
        mamba_state_backup_mode=args.mamba_state_backup_mode,
        use_target_kv_cache=not args.disable_target_kv_cache,
        use_mamba_cuda_graphs=args.mamba_use_cuda_graphs,
        mamba_cg_warmups=args.mamba_cg_warmups,
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