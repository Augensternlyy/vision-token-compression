from typing import Dict, Optional, Tuple

import math

import torch
import torch.nn.functional as F


def _visual_token_count(projected_features: torch.Tensor) -> int:
    if projected_features.ndim < 2:
        raise ValueError("projected_features must have at least 2 dimensions.")
    return int(projected_features.shape[-2])


def _index_select_visual_tokens(projected_features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    visual_dim = projected_features.ndim - 2
    return projected_features.index_select(visual_dim, indices)


def _selector_info(method: str, original_n: int, selected_n: int, indices, note: str) -> Dict:
    info = {
        "selector_method": method,
        "original_visual_tokens": int(original_n),
        "retained_visual_tokens": int(selected_n),
        "selector_note": note,
    }
    if indices is not None:
        info["selected_indices"] = [int(i) for i in indices[:20]]
        if len(indices) > 20:
            info["selected_indices_truncated"] = True
    return info


def _validate_retain_tokens(retain_tokens: Optional[int]) -> None:
    if retain_tokens is not None and retain_tokens <= 0:
        raise ValueError("retain_tokens must be positive when token selection is enabled.")


def _resolve_grid_size(n_tokens: int, image_grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if image_grid_size is not None:
        height, width = image_grid_size
        if height <= 0 or width <= 0:
            raise ValueError("image_grid_size must contain positive integers.")
        if height * width != n_tokens:
            raise ValueError(
                f"image_grid_size {image_grid_size} does not match {n_tokens} visual tokens."
            )
        return int(height), int(width)

    grid_size = int(math.isqrt(n_tokens))
    if grid_size * grid_size != n_tokens:
        raise ValueError(
            "spatial_uniform requires a square number of visual tokens or an explicit image_grid_size."
        )
    return grid_size, grid_size


def select_visual_tokens(
    projected_features,
    method: str,
    retain_tokens: Optional[int],
    seed: int = 42,
    question: Optional[str] = None,
    input_ids=None,
    tokenizer=None,
    image_grid_size: Optional[tuple] = None,
    selector_extra: Optional[dict] = None,
):
    if not torch.is_tensor(projected_features):
        raise TypeError("projected_features must be a torch.Tensor.")

    normalized_method = (method or "llava").lower()
    original_n = _visual_token_count(projected_features)

    if normalized_method in {"llava", "none", "vanilla"}:
        return projected_features, _selector_info(
            normalized_method,
            original_n,
            original_n,
            None,
            "No visual token selection was applied.",
        )

    if normalized_method == "visionzip":
        raise ValueError(
            "visionzip requires CLIP hidden states and attentions; use "
            "select_visionzip_visual_tokens before projection instead."
        )

    if normalized_method not in {"first_n", "random", "spatial_uniform"}:
        raise ValueError(f"Unknown visual token selection method: {method}")

    _validate_retain_tokens(retain_tokens)
    if retain_tokens is None or retain_tokens >= original_n:
        return projected_features, _selector_info(
            normalized_method,
            original_n,
            original_n,
            None,
            "retain_tokens is unset or not smaller than the original visual token count; no pruning applied.",
        )

    retain_tokens = int(retain_tokens)
    device = projected_features.device

    if normalized_method == "first_n":
        indices = torch.arange(retain_tokens, device=device, dtype=torch.long)
        selected = _index_select_visual_tokens(projected_features, indices)
        return selected, _selector_info(
            normalized_method,
            original_n,
            retain_tokens,
            indices.detach().cpu().tolist(),
            f"Kept the first {retain_tokens} visual tokens.",
        )

    if normalized_method == "random":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = torch.randperm(original_n, generator=generator, device=device)[:retain_tokens]
        indices = indices.sort().values
        selected = _index_select_visual_tokens(projected_features, indices)
        return selected, _selector_info(
            normalized_method,
            original_n,
            retain_tokens,
            indices.detach().cpu().tolist(),
            f"Randomly sampled {retain_tokens} visual tokens with seed {seed}, then restored index order.",
        )

    if normalized_method == "spatial_uniform":
        height, width = _resolve_grid_size(original_n, image_grid_size)
        sample_side = int(math.ceil(math.sqrt(retain_tokens)))
        row_coords = torch.linspace(0, height - 1, sample_side, device=device).round().long()
        col_coords = torch.linspace(0, width - 1, sample_side, device=device).round().long()
        rows, cols = torch.meshgrid(row_coords, col_coords, indexing="ij")
        indices = (rows.reshape(-1) * width + cols.reshape(-1)).unique(sorted=True)
        if indices.numel() < retain_tokens:
            extra = torch.arange(original_n, device=device, dtype=torch.long)
            missing = extra[~torch.isin(extra, indices)][: retain_tokens - indices.numel()]
            indices = torch.cat([indices, missing]).sort().values
        else:
            indices = indices[:retain_tokens].sort().values
        selected = _index_select_visual_tokens(projected_features, indices)
        return selected, _selector_info(
            normalized_method,
            original_n,
            int(indices.numel()),
            indices.detach().cpu().tolist(),
            f"Uniformly sampled {int(indices.numel())} tokens from a {height}x{width} visual grid.",
        )

    raise ValueError(f"Unknown visual token selection method: {method}")


def _visionzip_counts(retain_tokens: Optional[int], selector_extra: Optional[dict]) -> Tuple[int, int]:
    selector_extra = selector_extra or {}
    contextual = int(selector_extra.get("contextual", 10))
    if contextual < 0:
        raise ValueError("visionzip contextual must be non-negative.")

    if "dominant" in selector_extra:
        dominant = int(selector_extra["dominant"])
    elif retain_tokens is not None:
        dominant = int(retain_tokens) - contextual
    else:
        dominant = 54

    if dominant <= 0:
        raise ValueError("visionzip dominant must be positive.")
    return dominant, contextual


def select_visionzip_visual_tokens(
    clip_hidden_states: torch.Tensor,
    clip_attentions: torch.Tensor,
    projector,
    retain_tokens: Optional[int] = None,
    selector_extra: Optional[dict] = None,
):
    """VisionZip token selection before the multimodal projector.

    The implementation follows VisionZip's inference recipe: select dominant
    tokens by CLS attention, then summarize the remaining tokens into contextual
    tokens by similarity-based merging. The returned tensor is already projected
    to the LLM hidden size, matching LLaVA's normal image feature path.
    """
    if clip_hidden_states.ndim != 3:
        raise ValueError("clip_hidden_states must have shape [batch, tokens, hidden].")
    if clip_attentions.ndim != 4:
        raise ValueError("clip_attentions must have shape [batch, heads, tokens, tokens].")

    dominant, contextual = _visionzip_counts(retain_tokens, selector_extra)
    batch, total_with_cls, hidden = clip_hidden_states.shape
    original_patch_tokens = total_with_cls - 1
    if original_patch_tokens <= 0:
        raise ValueError("VisionZip requires CLIP patch tokens.")

    dominant = min(dominant, original_patch_tokens + 1)
    max_contextual = max(total_with_cls - dominant, 0)
    contextual = min(contextual, max_contextual)

    cls_attention = clip_attentions[:, :, 0, 1:].sum(dim=1).float()
    dominant_patch_count = max(dominant - 1, 0)
    if dominant_patch_count > 0:
        topk_patch_indices = cls_attention.topk(dominant_patch_count, dim=1).indices + 1
        cls_indices = torch.zeros((batch, 1), dtype=topk_patch_indices.dtype, device=topk_patch_indices.device)
        dominant_indices = torch.cat([cls_indices, topk_patch_indices], dim=1)
    else:
        dominant_indices = torch.zeros((batch, 1), dtype=torch.long, device=clip_hidden_states.device)

    dominant_tokens = torch.gather(
        clip_hidden_states,
        dim=1,
        index=dominant_indices.unsqueeze(-1).expand(-1, -1, hidden),
    )

    if contextual > 0:
        dominant_mask = torch.zeros((batch, total_with_cls), dtype=torch.bool, device=clip_hidden_states.device)
        dominant_mask.scatter_(1, dominant_indices, True)
        remaining_tokens = clip_hidden_states[~dominant_mask].view(batch, total_with_cls - dominant_indices.shape[1], hidden)

        metric = F.normalize(remaining_tokens.float(), dim=-1)
        step = max(1, metric.shape[1] // contextual)
        target_indices = torch.arange(0, metric.shape[1], step, device=metric.device)[:contextual]
        target_tokens = metric[:, target_indices, :]
        target_hidden = remaining_tokens[:, target_indices, :]

        non_target = ~torch.isin(torch.arange(metric.shape[1], device=metric.device), target_indices)
        tokens_to_merge = metric[:, non_target, :]
        hidden_to_merge = remaining_tokens[:, non_target, :]
        if tokens_to_merge.shape[1] > 0:
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assignments = similarity.argmax(dim=2)
            assign_one_hot = torch.zeros(
                tokens_to_merge.shape[0],
                tokens_to_merge.shape[1],
                contextual,
                dtype=remaining_tokens.dtype,
                device=remaining_tokens.device,
            )
            assign_one_hot.scatter_(2, assignments.unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            contextual_tokens = target_hidden + aggregated_hidden
        else:
            contextual_tokens = target_hidden
        selected_clip_tokens = torch.cat([dominant_tokens, contextual_tokens], dim=1)
    else:
        selected_clip_tokens = dominant_tokens

    if projector is not None:
        try:
            projector_param = next(projector.parameters())
            selected_clip_tokens = selected_clip_tokens.to(device=projector_param.device, dtype=projector_param.dtype)
        except StopIteration:
            pass
        projected = projector(selected_clip_tokens)
    else:
        projected = selected_clip_tokens
    selected_n = int(projected.shape[-2])
    info = _selector_info(
        "visionzip",
        original_patch_tokens,
        selected_n,
        None,
        f"VisionZip kept {dominant} dominant tokens and {contextual} contextual tokens.",
    )
    info["visionzip_dominant"] = int(dominant)
    info["visionzip_contextual"] = int(contextual)
    return projected, info
