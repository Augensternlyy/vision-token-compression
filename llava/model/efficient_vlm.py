from types import MethodType
from typing import Dict, Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    Cache,
    DynamicCache,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


SPARSEVLM_SCHEDULES = {
    "1_0": {
        192: [300, 200, 110],
        128: [303, 110, 36],
        96: [238, 48, 26],
        64: [66, 30, 17],
    },
    "2_0": {
        192: [300, 200, 118],
        128: [238, 108, 60],
        96: [246, 54, 28],
        64: [66, 34, 20],
    },
}


def _prepare_attention_mask(model, attention_mask_2d, batch_size, seq_length, inputs_embeds, past_key_values_length, output_attentions):
    if getattr(model, "_use_flash_attention_2", False):
        return attention_mask_2d if (attention_mask_2d is not None and 0 in attention_mask_2d) else None
    if getattr(model, "_use_sdpa", False) and not output_attentions:
        return _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask_2d,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    return _prepare_4d_causal_attention_mask(
        attention_mask_2d,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )


def _rebuild_prefill_attention_mask(model, batch_size, hidden_states, position_ids, output_attentions):
    seq_length = hidden_states.shape[1]
    attention_mask_2d = torch.ones(
        (batch_size, seq_length),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    return _prepare_attention_mask(
        model,
        attention_mask_2d,
        batch_size,
        seq_length,
        hidden_states,
        0,
        output_attentions,
    )


def _keep_by_indices(hidden_states, position_ids, keep_indices):
    keep_indices = keep_indices.sort().values
    hidden_states = hidden_states.index_select(1, keep_indices)
    if position_ids is not None:
        position_ids = torch.arange(
            hidden_states.shape[1],
            dtype=position_ids.dtype,
            device=hidden_states.device,
        ).unsqueeze(0)
    return hidden_states, position_ids, keep_indices


def _fastv_keep_indices(attention, image_start: int, image_len: int, seq_len: int, config: Dict, device):
    retain_tokens = config.get("retain_tokens")
    if retain_tokens is not None:
        keep_image = int(retain_tokens)
    else:
        prune_ratio = float(config.get("r", config.get("fastv_r", 0.5)))
        keep_image = round(image_len * (1.0 - prune_ratio))
    keep_image = max(1, min(int(keep_image), int(image_len)))

    image_scores = attention.mean(dim=1)[0, -1, image_start : image_start + image_len]
    top_image = image_scores.topk(keep_image).indices + image_start
    return torch.cat(
        [
            torch.arange(image_start, device=device),
            top_image,
            torch.arange(image_start + image_len, seq_len, device=device),
        ]
    )


def _sparsevlm_schedule(config: Dict, retain_tokens: Optional[int]):
    if "schedule" in config:
        return [int(x) for x in config["schedule"]]
    version = str(config.get("version", "1_0"))
    schedules = SPARSEVLM_SCHEDULES.get(version, SPARSEVLM_SCHEDULES["1_0"])
    retain = int(config.get("retain_tokens", retain_tokens or 192))
    if retain not in schedules:
        return [retain, retain, retain]
    return schedules[retain]


def _select_sparsevlm_text_indices(hidden_states, image_start: int, image_len: int):
    text_start = image_start + image_len
    if text_start >= hidden_states.shape[1]:
        return None
    visual_tokens = hidden_states[:, image_start:text_start, :]
    text_tokens = hidden_states[:, text_start:, :]
    if text_tokens.shape[1] == 0:
        return None
    scores = torch.matmul(visual_tokens, text_tokens.transpose(1, 2)).softmax(dim=2).mean(dim=1)
    selected = torch.where(scores > scores.mean())
    if selected[1].numel() == 0:
        return torch.tensor([text_tokens.shape[1] - 1], device=hidden_states.device)
    return selected[1]


def _sparsevlm_keep_indices(attention, image_start: int, image_len: int, seq_len: int, text_indices, keep_image: int, device):
    text_start = image_start + image_len
    keep_image = max(1, min(int(keep_image), int(image_len)))
    if text_indices is None or text_indices.numel() == 0:
        query_indices = torch.tensor([seq_len - 1], device=device)
    else:
        query_indices = torch.clamp(text_indices + text_start, max=seq_len - 1)

    weights = attention.mean(dim=1)
    image_scores = weights[:, query_indices, image_start:text_start].mean(dim=1)[0]
    top_image = image_scores.topk(keep_image).indices + image_start
    return torch.cat(
        [
            torch.arange(image_start, device=device),
            top_image,
            torch.arange(text_start, seq_len, device=device),
        ]
    )


def _efficient_llama_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    config = getattr(self, "_efficient_vlm_config", None) or {}
    method = config.get("method")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    past_key_values_length = 0
    use_legacy_cache = False
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    attention_mask = _prepare_attention_mask(
        self,
        attention_mask,
        batch_size,
        seq_length,
        inputs_embeds,
        past_key_values_length,
        output_attentions,
    )

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    image_start = getattr(self, "_efficient_image_token_start", None)
    image_len = getattr(self, "_efficient_image_token_length", None)
    can_prune = (
        method in {"fastv", "sparsevlm"}
        and image_start is not None
        and image_len is not None
        and hidden_states.shape[1] > 1
        and past_key_values_length == 0
        and batch_size == 1
    )
    image_start = int(image_start or 0)
    image_len = int(image_len or 0)
    last_attention = None
    sparse_text_indices = _select_sparsevlm_text_indices(hidden_states, image_start, image_len) if can_prune and method == "sparsevlm" else None
    sparse_layers = [int(x) for x in config.get("layers", [2, 6, 15])]
    sparse_schedule = _sparsevlm_schedule(config, config.get("retain_tokens"))

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if can_prune and method == "fastv":
            k_layer = int(config.get("k", config.get("fastv_k", 3)))
            if layer_idx == k_layer and last_attention is not None:
                keep_indices = _fastv_keep_indices(
                    last_attention,
                    image_start,
                    image_len,
                    hidden_states.shape[1],
                    config,
                    hidden_states.device,
                )
                hidden_states, position_ids, keep_indices = _keep_by_indices(hidden_states, position_ids, keep_indices)
                image_len = int(((keep_indices >= image_start) & (keep_indices < image_start + image_len)).sum().item())
                attention_mask = _rebuild_prefill_attention_mask(self, batch_size, hidden_states, position_ids, output_attentions)

        force_attentions = False
        if can_prune and method == "fastv":
            force_attentions = layer_idx == int(config.get("k", config.get("fastv_k", 3))) - 1
        if can_prune and method == "sparsevlm":
            force_attentions = layer_idx in sparse_layers

        layer_attention_mask = attention_mask
        layer_position_ids = position_ids
        if method in {"fastv", "sparsevlm"} and past_key_values_length > 0 and seq_length == 1 and isinstance(past_key_values, Cache):
            layer_past_length = past_key_values.get_usable_length(seq_length, layer_idx)
            layer_position_ids = torch.full(
                (batch_size, seq_length),
                layer_past_length,
                dtype=position_ids.dtype,
                device=hidden_states.device,
            )
            layer_attention_mask = None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=layer_position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions or force_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]
        layer_attention = layer_outputs[1] if (output_attentions or force_attentions) else None

        if can_prune and method == "fastv" and force_attentions:
            last_attention = layer_attention

        if can_prune and method == "sparsevlm" and layer_idx in sparse_layers and layer_attention is not None:
            schedule_idx = min(sparse_layers.index(layer_idx), len(sparse_schedule) - 1)
            keep_indices = _sparsevlm_keep_indices(
                layer_attention,
                image_start,
                image_len,
                hidden_states.shape[1],
                sparse_text_indices,
                sparse_schedule[schedule_idx],
                hidden_states.device,
            )
            hidden_states, position_ids, keep_indices = _keep_by_indices(hidden_states, position_ids, keep_indices)
            image_len = int(((keep_indices >= image_start) & (keep_indices < image_start + image_len)).sum().item())
            attention_mask = _rebuild_prefill_attention_mask(self, batch_size, hidden_states, position_ids, output_attentions)

        if use_cache:
            next_decoder_cache = layer_outputs[2 if (output_attentions or force_attentions) else 1]

        if output_attentions:
            all_self_attns += (layer_attention,)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def enable_efficient_llama(model, method: str, config: Optional[Dict] = None):
    base = model.get_model() if hasattr(model, "get_model") else model
    if not hasattr(base, "_original_llama_forward"):
        base._original_llama_forward = base.forward
        base.forward = MethodType(_efficient_llama_forward, base)
    base._efficient_vlm_config = {"method": method, **(config or {})}
    return model


def disable_efficient_llama(model):
    base = model.get_model() if hasattr(model, "get_model") else model
    if hasattr(base, "_original_llama_forward"):
        base.forward = base._original_llama_forward
        delattr(base, "_original_llama_forward")
    if hasattr(base, "_efficient_vlm_config"):
        delattr(base, "_efficient_vlm_config")
    return model


def set_image_token_span(model, image_start: int, image_length: int):
    base = model.get_model() if hasattr(model, "get_model") else model
    base._efficient_image_token_start = int(image_start)
    base._efficient_image_token_length = int(image_length)
