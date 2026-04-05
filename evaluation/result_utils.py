import json
import os
import re
import time
from typing import Any, Dict, Optional


def _safe_name(value: Optional[str], fallback: str = "unknown") -> str:
    if not value:
        return fallback
    cleaned = str(value).strip().replace("\\", "/")
    cleaned = cleaned.split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def infer_task_type(question_file: Optional[str], bench_name: Optional[str]) -> str:
    if question_file:
        return _safe_name(os.path.splitext(os.path.basename(question_file))[0], fallback="task")
    return _safe_name(bench_name or "task", fallback="task")


def build_result_bundle(
    model_path: str,
    drafter_path: str,
    draft_tokens: Optional[int],
    question_file: Optional[str],
    bench_name: Optional[str],
    version: str,
    draft_model_label: Optional[str] = None,
    draft_bucket_label: Optional[str] = None,
    answer_file_override: Optional[str] = None,
) -> Dict[str, str]:
    if answer_file_override:
        answer_file = answer_file_override
        result_dir = os.path.dirname(answer_file) or "."
    else:
        base_model = _safe_name(model_path)
        draft_model = _safe_name(draft_model_label if draft_model_label is not None else drafter_path)
        if draft_bucket_label is not None:
            draft_bucket = _safe_name(draft_bucket_label)
        else:
            draft_bucket = "adaptive" if draft_tokens is None else str(int(draft_tokens))
        task_type = infer_task_type(question_file, bench_name)
        version_name = _safe_name(version, fallback="run")
        result_dir = os.path.join(
            "results",
            base_model,
            draft_model,
            draft_bucket,
            task_type,
            version_name,
        )
        answer_file = os.path.join(result_dir, "answers.jsonl")

    return {
        "result_dir": result_dir,
        "answer_file": answer_file,
        "config_file": os.path.join(result_dir, "run_config.json"),
        "metrics_file": os.path.join(result_dir, "metrics.json"),
    }


def write_run_config(config_file: str, args_dict: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": args_dict,
    }
    if extra:
        payload.update(extra)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def summarize_answer_file(answer_file: str, draft_tokens: Optional[int]) -> Dict[str, Any]:
    total_questions = 0
    total_choices = 0
    total_turns = 0
    total_new_tokens = 0
    total_wall_time = 0.0
    total_steps = 0
    accept_lengths = []

    if not os.path.exists(answer_file):
        return {
            "error": f"Answer file not found: {answer_file}",
        }

    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_questions += 1
            choices = row.get("choices", [])
            total_choices += len(choices)

            for choice in choices:
                new_tokens = choice.get("new_tokens", [])
                wall_time = choice.get("wall_time", [])
                decoding_steps = choice.get("decoding_steps", [])
                cur_accept = choice.get("accept_lengths", [])

                total_turns += len(new_tokens)
                total_new_tokens += int(sum(new_tokens))
                total_wall_time += float(sum(wall_time))
                total_steps += int(sum(decoding_steps))
                accept_lengths.extend(cur_accept)

    throughput = (total_new_tokens / total_wall_time) if total_wall_time > 0 else None
    avg_time_per_token = (total_wall_time / total_new_tokens) if total_new_tokens > 0 else None
    avg_accept_tokens_per_step = (
        float(sum(accept_lengths)) / len(accept_lengths) if len(accept_lengths) > 0 else None
    )

    acceptance_rate = None
    if draft_tokens is not None and draft_tokens > 0 and len(accept_lengths) > 0:
        accepted_draft_tokens = sum(max(int(x) - 1, 0) for x in accept_lengths)
        proposed_draft_tokens = int(draft_tokens) * len(accept_lengths)
        acceptance_rate = accepted_draft_tokens / proposed_draft_tokens if proposed_draft_tokens > 0 else None

    return {
        "total_questions": total_questions,
        "total_choices": total_choices,
        "total_turns": total_turns,
        "total_new_tokens": total_new_tokens,
        "total_wall_time_sec": total_wall_time,
        "total_decoding_steps": total_steps,
        "throughput_tokens_per_sec": throughput,
        "avg_time_per_token_sec": avg_time_per_token,
        "avg_accept_tokens_per_step": avg_accept_tokens_per_step,
        "acceptance_rate": acceptance_rate,
        "acceptance_rate_note": (
            "Estimated as accepted_draft_tokens/proposed_draft_tokens and only available for fixed draft_tokens."
            if draft_tokens is not None
            else "Not computed for adaptive draft_tokens because proposed draft count varies by step."
        ),
    }


def write_metrics(metrics_file: str, metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
