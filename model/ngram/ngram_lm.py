import gzip
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

Context = Tuple[int, ...]


def _open_maybe_gzip(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def iter_corpus_texts(corpus_path: str, jsonl_text_field: str = "text") -> Iterable[str]:
    """Yield training texts from raw text or JSONL corpora."""
    with open(corpus_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            if line.startswith("{"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue

                if isinstance(record, dict):
                    text_value = record.get(jsonl_text_field)
                    if isinstance(text_value, str) and text_value:
                        yield text_value
                        continue

                    turns_value = record.get("turns")
                    if isinstance(turns_value, list):
                        for turn in turns_value:
                            if isinstance(turn, str) and turn:
                                yield turn
                        continue
                continue

            yield line


class TokenNGramModel:
    """A token-level n-gram LM with greedy backoff decoding."""

    def __init__(self, order: int = 4):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.context_next_counts: DefaultDict[Context, Counter] = defaultdict(Counter)
        self.unigram_counts: Counter = Counter()
        self.total_tokens = 0

    def update(self, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return

        history: List[int] = []
        for token in token_ids:
            token = int(token)
            self.unigram_counts[token] += 1
            self.total_tokens += 1

            max_context_len = min(self.order - 1, len(history))
            for context_len in range(0, max_context_len + 1):
                context = tuple(history[-context_len:]) if context_len > 0 else ()
                self.context_next_counts[context][token] += 1

            history.append(token)

    def predict_next_token(self, history: Sequence[int]) -> Optional[int]:
        if not self.unigram_counts:
            return None

        max_context_len = min(self.order - 1, len(history))
        for context_len in range(max_context_len, -1, -1):
            context = tuple(history[-context_len:]) if context_len > 0 else ()
            next_counts = self.context_next_counts.get(context)
            if next_counts:
                return next_counts.most_common(1)[0][0]

        return self.unigram_counts.most_common(1)[0][0]

    def predict_tokens(self, history: Sequence[int], num_tokens: int) -> List[int]:
        if num_tokens <= 0:
            return []

        working_history = [int(token) for token in history]
        predictions: List[int] = []

        for _ in range(num_tokens):
            token = self.predict_next_token(working_history)
            if token is None:
                break
            predictions.append(token)
            working_history.append(token)

        return predictions

    def state_dict(self) -> Dict[str, object]:
        return {
            "order": self.order,
            "total_tokens": self.total_tokens,
            "unigram_counts": dict(self.unigram_counts),
            "context_next_counts": {
                context: dict(counter) for context, counter in self.context_next_counts.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "TokenNGramModel":
        model = cls(order=int(state["order"]))
        model.total_tokens = int(state.get("total_tokens", 0))
        model.unigram_counts = Counter({int(k): int(v) for k, v in state["unigram_counts"].items()})

        context_map = state.get("context_next_counts", {})
        reconstructed: DefaultDict[Context, Counter] = defaultdict(Counter)
        for context, counts in context_map.items():
            # Context can come from pickle as tuple[int, ...] or from a JSON-like payload.
            if isinstance(context, tuple):
                normalized_context = tuple(int(token) for token in context)
            else:
                normalized_context = tuple(int(token) for token in list(context))
            reconstructed[normalized_context] = Counter({int(k): int(v) for k, v in counts.items()})

        model.context_next_counts = reconstructed
        return model

    def save(self, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with _open_maybe_gzip(output_path, "wb") as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, input_path: str) -> "TokenNGramModel":
        with _open_maybe_gzip(input_path, "rb") as f:
            state = pickle.load(f)
        return cls.from_state_dict(state)


def train_token_ngram_model(
    tokenizer,
    corpus_path: str,
    order: int,
    max_texts: Optional[int],
    jsonl_text_field: str,
) -> Tuple[TokenNGramModel, Dict[str, int]]:
    model = TokenNGramModel(order=order)
    text_count = 0
    token_count = 0

    for text in iter_corpus_texts(corpus_path, jsonl_text_field=jsonl_text_field):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue

        model.update(token_ids)
        text_count += 1
        token_count += len(token_ids)

        if max_texts is not None and text_count >= max_texts:
            break

    if model.total_tokens == 0:
        raise ValueError(
            f"Could not train n-gram model from '{corpus_path}'. No tokenized text was found."
        )

    stats = {
        "texts": text_count,
        "tokens": token_count,
        "order": order,
    }
    return model, stats
