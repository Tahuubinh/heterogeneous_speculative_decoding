import copy
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import _crop_past_key_values

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer


@torch.no_grad()
def find_candidate_pred_tokens_from_ngram_lm(input_ids, ngram_lm, num_pred_tokens=10):
    if ngram_lm is None:
        raise ValueError("ngram_lm must be provided")
    if num_pred_tokens <= 0:
        raise ValueError("num_pred_tokens must be > 0")

    history = input_ids[0].tolist()
    pred_tokens = ngram_lm.predict_tokens(history, num_pred_tokens)
    if not pred_tokens:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    return torch.tensor(pred_tokens, dtype=torch.long, device=input_ids.device)


@torch.no_grad()
def greedy_search_ngram(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    draft_num_candidate_tokens=10,
    fallback_token_id=100,
    ngram_lm=None,
    **model_kwargs,
):
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    if eos_token_id is not None:
        eos_token_id_tensor = torch.tensor(eos_token_id, device=input_ids.device)
    else:
        eos_token_id_tensor = None

    scores = () if (return_dict_in_generate and output_scores) else None
    max_len = stopping_criteria[0].max_length

    step = 0
    accept_length_list = []

    while True:
        step += 1
        cur_len = input_ids.shape[-1]

        candidate_pred_tokens = find_candidate_pred_tokens_from_ngram_lm(
            input_ids,
            ngram_lm,
            draft_num_candidate_tokens,
        )

        if len(candidate_pred_tokens) == 0:
            candidate_pred_tokens = torch.tensor(
                [fallback_token_id],
                device=input_ids.device,
                dtype=input_ids.dtype,
            ).unsqueeze(0)
        else:
            candidate_pred_tokens = candidate_pred_tokens.unsqueeze(0)

        candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)
        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

        candidate_kwargs = copy.copy(model_kwargs)
        attention_mask = candidate_kwargs["attention_mask"]
        mask_extension_length = candidate_input_ids.shape[1] - attention_mask.shape[1]
        candidate_kwargs["attention_mask"] = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], mask_extension_length)),
            ],
            dim=-1,
        )

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        new_logits = outputs.logits[:, -candidate_length - 1 :]
        selected_tokens = new_logits.argmax(dim=-1)
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
        n_matches = min(n_matches, max_len - cur_len - 1)

        valid_tokens = selected_tokens[:, : n_matches + 1]
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        new_cur_len = input_ids.shape[-1]

        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cur_len - 1)
        model_kwargs["past_key_values"] = outputs.past_key_values

        accept_length = new_cur_len - cur_len
        accept_length_list.append(accept_length)

        if eos_token_id_tensor is not None and torch.isin(valid_tokens, eos_token_id_tensor).any():
            break

        if stopping_criteria(input_ids, scores):
            break

    idx = step - 1
    return input_ids, idx, accept_length_list
