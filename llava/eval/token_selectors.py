from typing import Dict, Optional, Tuple

import math

import torch


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
