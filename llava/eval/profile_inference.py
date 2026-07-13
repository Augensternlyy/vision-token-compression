import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.eval.token_selectors import select_visual_tokens
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")

def split_image_files(image_file: str, sep: str) -> List[str]:
    return [x for x in image_file.split(sep) if x]


def infer_conv_mode(model_name: str) -> str:
    lower_name = model_name.lower()
    if "llama-2" in lower_name:
        return "llava_llama_2"
    if "mistral" in lower_name:
        return "mistral_instruct"
    if "v1.6-34b" in lower_name:
        return "chatml_direct"
    if "v1" in lower_name:
        return "llava_v1"
    if "mpt" in lower_name:
        return "mpt"
    return "llava_v0"


def build_prompt(query: str, model: nn.Module, conv_mode: str) -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if getattr(model.config, "mm_use_im_start_end", False):
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def cuda_available() -> bool:
    return torch.cuda.is_available()

def sync_cuda() -> None:
    if cuda_available():
        torch.cuda.synchronize()


def bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024.0**2)


def measure_wall_ms(fn):
    sync_cuda()
    start = time.perf_counter()
    out = fn()
    sync_cuda()
    return out, (time.perf_counter() - start) * 1000.0


def measure_gpu_ms(fn):
    if not cuda_available():
        return measure_wall_ms(fn)
    sync_cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)
def measure_gpu_ms(fn):
    if not cuda_available():
        return measure_wall_ms(fn)
    sync_cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)

def tensor_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def move_images_to_device(images: Any, device: torch.device, dtype: torch.dtype):
    if isinstance(images, list):
        return [image.to(device=device, dtype=dtype) for image in images]
    return images.to(device=device, dtype=dtype)


def concat_images_for_vision(images: Any) -> torch.Tensor:
    if isinstance(images, list):
        normalized = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
        return torch.cat(normalized, dim=0)
    if torch.is_tensor(images) and images.ndim == 5:
        return torch.cat([image for image in images], dim=0)
    return images


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: Optional[float],
) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class NamedModuleTimer:
    def __init__(self, model: nn.Module, keywords: Iterable[str]):
        self.keywords = tuple(k.lower() for k in keywords)
        self.handles = []
        self.events = []
        self.wall_ms = 0.0
        self.module_names = []

        for name, module in model.named_modules():
            lname = name.lower()
            if not name or not any(k in lname for k in self.keywords):
                continue
            self.module_names.append(name)
            self.handles.append(module.register_forward_pre_hook(self._pre_hook))
            self.handles.append(module.register_forward_hook(self._post_hook))

    def _pre_hook(self, module, inputs):
        if cuda_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            module.__profile_start_event = event
        else:
            module.__profile_start_wall = time.perf_counter()

    def _post_hook(self, module, inputs, output):
        if cuda_available():
            start = getattr(module, "__profile_start_event", None)
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.events.append((start, end))
        else:
            start = getattr(module, "__profile_start_wall", None)
            if start is not None:
                self.wall_ms += (time.perf_counter() - start) * 1000.0

    def total_ms(self) -> Optional[float]:
        if not self.module_names:
            return None
        if cuda_available():
            torch.cuda.synchronize()
            return sum(start.elapsed_time(end) for start, end in self.events)
        return self.wall_ms

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def linear_module_flops(module: nn.Module, leading_tokens: int) -> int:
    if isinstance(module, nn.Linear):
        return int(2 * leading_tokens * module.in_features * module.out_features)
    return 0


def estimate_projector_flops(projector: nn.Module, token_count: int) -> Optional[int]:
    if projector is None or token_count <= 0:
        return None
    return sum(linear_module_flops(module, token_count) for module in projector.modules())


def estimate_vit_flops(vision_tower: nn.Module, images: torch.Tensor) -> Optional[int]:
    cfg = getattr(vision_tower, "config", None)
    if cfg is None:
        return None

    hidden = getattr(cfg, "hidden_size", None)
    intermediate = getattr(cfg, "intermediate_size", None)
    layers = getattr(cfg, "num_hidden_layers", None)
    patch = getattr(cfg, "patch_size", None)
    channels = getattr(cfg, "num_channels", 3)
    if any(v is None for v in (hidden, intermediate, layers, patch)):
        return None

    if images.ndim == 5:
        batch = images.shape[0] * images.shape[1]
        height, width = images.shape[-2:]
    elif images.ndim == 4:
        batch = images.shape[0]
        height, width = images.shape[-2:]
    else:
        return None

    patches = (height // patch) * (width // patch)
    seq_len = patches + 1
    patch_embed = 2 * patches * patch * patch * channels * hidden
    per_layer = seq_len * (8 * hidden * hidden + 4 * hidden * intermediate) + 4 * seq_len * seq_len * hidden
    return int(batch * (patch_embed + layers * per_layer))


def estimate_llm_flops(
    config,
    prefill_tokens: int,
    decode_forward_tokens: int,
    generated_tokens: int,
    include_lm_head: bool,
) -> Optional[Dict[str, int]]:
    hidden = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    intermediate = getattr(config, "intermediate_size", None)
    vocab = getattr(config, "vocab_size", None)
    if any(v is None for v in (hidden, layers, intermediate)):
        return None

    def block_flops(tokens: int, kv_tokens: int) -> int:
        projections_and_mlp = tokens * (8 * hidden * hidden + 6 * hidden * intermediate)
        attention = 4 * tokens * kv_tokens * hidden
        return int(layers * (projections_and_mlp + attention))

    prefill = block_flops(prefill_tokens, prefill_tokens)
    if include_lm_head and vocab is not None:
        prefill += int(2 * prefill_tokens * hidden * vocab)

    decode = 0
    for i in range(decode_forward_tokens):
        kv_tokens = prefill_tokens + 1 + i
        decode += block_flops(1, kv_tokens)
        if include_lm_head and vocab is not None:
            decode += int(2 * hidden * vocab)

    return {
        "llm_prefill_flops": int(prefill),
        "llm_decode_flops": int(decode),
        "llm_total_flops": int(prefill + decode),
        "generated_tokens_for_flops": int(generated_tokens),
    }


def get_projector(model: nn.Module) -> Optional[nn.Module]:
    base = model.get_model() if hasattr(model, "get_model") else model
    return getattr(base, "mm_projector", None)


def is_hf_llava_model_path(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == "llava" and "text_config" in config


def resolve_profile_model_name(model_path: str, fallback_model_name: str) -> str:
    if "llava" in fallback_model_name.lower():
        return fallback_model_name

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return fallback_model_name
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback_model_name

    candidates = [
        config.get("_name_or_path"),
        config.get("model_type"),
        *(config.get("architectures") or []),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and "llava" in candidate.lower():
            return candidate
    return fallback_model_name


def is_hf_llava_model(model: nn.Module) -> bool:
    return (
        getattr(getattr(model, "config", None), "model_type", None) == "llava"
        and hasattr(model, "vision_tower")
        and hasattr(model, "multi_modal_projector")
        and hasattr(model, "language_model")
    )


def load_profile_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device="cuda",
):
    model_name = resolve_profile_model_name(model_path, model_name)
    if not is_hf_llava_model_path(model_path):
        return load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=device,
        )

    if model_base is not None:
        raise ValueError(
            "The detected model is a HuggingFace LlavaForConditionalGeneration checkpoint "
            "and should be loaded without --model-base. For original LLaVA checkpoints, "
            "use a model whose config.json has model_type 'llava_llama'."
        )

    from transformers import AutoImageProcessor, AutoTokenizer, BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaProcessor

    kwargs = {"low_cpu_mem_usage": True}
    if device != "cuda":
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = "auto"

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)
    context_len = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 2048)
    return processor, model, image_processor, context_len


def prepare_multimodal_embeds_with_cached_images(
    model: nn.Module,
    input_ids: torch.Tensor,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    projected_features: torch.Tensor,
):
    original_encode_images = model.encode_images

    def cached_encode_images(_images):
        return projected_features

    model.encode_images = cached_encode_images
    try:
        return model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            images_tensor,
            image_sizes=image_sizes,
        )
    finally:
        model.encode_images = original_encode_images


@dataclass
class ProfileResult:
    image_preprocessing_time_ms: float
    vit_encoding_time_ms: Optional[float]
    projector_time_ms: Optional[float]
    selector_router_time_ms: Optional[float]
    selector_info: Optional[Dict[str, Any]]
    llm_prefill_time_ms: float
    first_token_latency_ttft_ms: float
    decode_time_ms: float
    total_latency_ms: float
    peak_gpu_memory_mib: Optional[float]
    peak_gpu_memory_incremental_mib: Optional[float]
    kv_cache_memory_mib: Optional[float]
    flops: Dict[str, Optional[int]]
    visual_token_count: int
    prompt_token_count: int
    prefill_token_count: int
    generated_token_count: int
    output_text: str


def build_hf_prompt(query: str, processor, image_count: int) -> str:
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": query})
    conversation = [{"role": "user", "content": content}]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:
            pass
    return ("<image>\n" * image_count) + f"USER: {query} ASSISTANT:"


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


@torch.inference_mode()
def profile_hf_once(args, processor, model) -> ProfileResult:
    images = [load_image(path) for path in split_image_files(args.image_file, args.sep)]
    model_device = getattr(model, "device", torch.device(args.device))
    model_dtype = next(model.parameters()).dtype
    prompt = build_hf_prompt(args.query, processor, len(images))

    if cuda_available():
        torch.cuda.empty_cache()
        base_gpu_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        base_gpu_memory = None

    latency_start = time.perf_counter()

    def preprocess():
        batch = processor(
            text=prompt,
            images=images if len(images) > 1 else images[0],
            return_tensors="pt",
        )
        return move_batch_to_device(batch, model_device, model_dtype)

    inputs, image_preprocessing_time_ms = measure_wall_ms(preprocess)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    def prefill():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=True,
            return_dict=True,
        )

    prefill_outputs, llm_prefill_time_ms = measure_gpu_ms(prefill)
    next_token = sample_next_token(
        prefill_outputs.logits[:, -1, :],
        args.temperature,
        args.top_p,
    )
    sync_cuda()
    first_token_latency_ttft_ms = (time.perf_counter() - latency_start) * 1000.0

    generated_ids = [next_token]
    past_key_values = prefill_outputs.past_key_values
    prefill_token_count = int(past_key_values[0][0].shape[2])
    image_token_index = getattr(model.config, "image_token_index", None)
    if image_token_index is None:
        prompt_token_count = int(input_ids.shape[1])
    else:
        prompt_token_count = int((input_ids[0] != image_token_index).sum().item())
    visual_token_count = max(prefill_token_count - prompt_token_count, 0)

    eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)
    decode_start = time.perf_counter()
    decode_forward_tokens = 0

    for _ in range(max(args.max_new_tokens - 1, 0)):
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

        step_attention_mask = torch.ones(
            (next_token.shape[0], prefill_token_count + len(generated_ids)),
            dtype=torch.long,
            device=next_token.device,
        )

        def decode_step():
            return model(
                input_ids=next_token,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        step_outputs, _step_ms = measure_gpu_ms(decode_step)
        decode_forward_tokens += 1
        past_key_values = step_outputs.past_key_values
        next_token = sample_next_token(
            step_outputs.logits[:, -1, :],
            args.temperature,
            args.top_p,
        )
        generated_ids.append(next_token)

    sync_cuda()
    decode_time_ms = (time.perf_counter() - decode_start) * 1000.0
    total_latency_ms = (time.perf_counter() - latency_start) * 1000.0

    generated = torch.cat(generated_ids, dim=1)
    tokenizer = getattr(processor, "tokenizer", processor)
    output_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    generated_token_count = int(generated.shape[1])

    if cuda_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated()
        peak_gpu_memory_incremental = peak_gpu_memory - base_gpu_memory
    else:
        peak_gpu_memory = None
        peak_gpu_memory_incremental = None

    kv_cache_memory = tensor_bytes(past_key_values)
    vit_flops = estimate_vit_flops(model.vision_tower, pixel_values) if pixel_values is not None else None
    projector_flops = estimate_projector_flops(model.multi_modal_projector, visual_token_count)
    llm_config = getattr(model.config, "text_config", model.config)
    llm_flops = estimate_llm_flops(
        llm_config,
        prefill_token_count,
        decode_forward_tokens,
        generated_token_count,
        args.include_lm_head_flops,
    ) or {}
    total_flops = sum(
        value
        for value in (vit_flops, projector_flops, llm_flops.get("llm_total_flops"))
        if value is not None
    )

    return ProfileResult(
        image_preprocessing_time_ms=image_preprocessing_time_ms,
        vit_encoding_time_ms=None,
        projector_time_ms=None,
        selector_router_time_ms=None,
        selector_info={
            "selector_method": args.method,
            "original_visual_tokens": visual_token_count,
            "retained_visual_tokens": visual_token_count,
            "selector_note": "HF LlavaForConditionalGeneration path does not expose projected visual tokens; selection was not applied.",
        },
        llm_prefill_time_ms=llm_prefill_time_ms,
        first_token_latency_ttft_ms=first_token_latency_ttft_ms,
        decode_time_ms=decode_time_ms,
        total_latency_ms=total_latency_ms,
        peak_gpu_memory_mib=bytes_to_mib(peak_gpu_memory),
        peak_gpu_memory_incremental_mib=bytes_to_mib(peak_gpu_memory_incremental),
        kv_cache_memory_mib=bytes_to_mib(kv_cache_memory),
        flops={
            "vit_flops": vit_flops,
            "projector_flops": projector_flops,
            **llm_flops,
            "total_estimated_flops": int(total_flops) if total_flops else None,
        },
        visual_token_count=visual_token_count,
        prompt_token_count=prompt_token_count,
        prefill_token_count=prefill_token_count,
        generated_token_count=generated_token_count,
        output_text=output_text,
    )


@torch.inference_mode()
def profile_once(args, tokenizer, model, image_processor) -> ProfileResult:
    if is_hf_llava_model(model):
        return profile_hf_once(args, tokenizer, model)

    images = [load_image(path) for path in split_image_files(args.image_file, args.sep)]
    image_sizes = [image.size for image in images]
    model_device = getattr(model, "device", torch.device(args.device))
    prompt = build_prompt(args.query, model, args.conv_mode)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model_device)

    if cuda_available():
        torch.cuda.empty_cache()
        base_gpu_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        base_gpu_memory = None

    latency_start = time.perf_counter()

    def preprocess():
        image_tensor = process_images(images, image_processor, model.config)
        return move_images_to_device(image_tensor, model_device, torch.float16)

    images_tensor, image_preprocessing_time_ms = measure_wall_ms(preprocess)

    vision_tower = model.get_vision_tower()
    projector = get_projector(model)
    module_timer = NamedModuleTimer(model, ("selector", "router"))

    try:
        vision_images = concat_images_for_vision(images_tensor)
        image_features, vit_encoding_time_ms = measure_gpu_ms(lambda: vision_tower(vision_images))
        if projector is not None:
            projected_features, projector_time_ms = measure_gpu_ms(lambda: projector(image_features))
        else:
            projected_features = image_features
            projector_time_ms = None

        projector_output_token_count = int(projected_features.numel() // projected_features.shape[-1])
        selector_extra = json.loads(args.selector_extra) if args.selector_extra else {}
        selector_output, selector_router_time_ms = measure_gpu_ms(
            lambda: select_visual_tokens(
                projected_features=projected_features,
                method=args.method,
                retain_tokens=args.retain_tokens,
                seed=args.seed,
                question=args.query,
                input_ids=input_ids,
                tokenizer=tokenizer,
                image_grid_size=None,
                selector_extra=selector_extra,
            )
        )
        projected_features, selector_info = selector_output

        prepared, _prepare_ms = measure_wall_ms(
            lambda: prepare_multimodal_embeds_with_cached_images(
                model,
                input_ids,
                images_tensor,
                image_sizes,
                projected_features,
            )
        )
        _input_ids, position_ids, attention_mask, _past, inputs_embeds, _labels = prepared
        if inputs_embeds is None:
            inputs_embeds = model.get_model().embed_tokens(input_ids)

        prefill_token_count = int(inputs_embeds.shape[1])
        prompt_token_count = int((input_ids[0] != IMAGE_TOKEN_INDEX).sum().item())
        visual_token_count = max(prefill_token_count - prompt_token_count, 0)

        def prefill():
            return model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

        prefill_outputs, llm_prefill_time_ms = measure_gpu_ms(prefill)
        next_token = sample_next_token(
            prefill_outputs.logits[:, -1, :],
            args.temperature,
            args.top_p,
        )
        sync_cuda()
        first_token_latency_ttft_ms = (time.perf_counter() - latency_start) * 1000.0

        generated_ids = [next_token]
        past_key_values = prefill_outputs.past_key_values
        eos_token_id = tokenizer.eos_token_id
        decode_start = time.perf_counter()
        decode_forward_tokens = 0

        for _ in range(max(args.max_new_tokens - 1, 0)):
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

            past_len = prefill_token_count + len(generated_ids) - 1
            step_position_ids = torch.full(
                (next_token.shape[0], 1),
                past_len,
                dtype=torch.long,
                device=next_token.device,
            )

            def decode_step():
                return model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    position_ids=step_position_ids,
                    use_cache=True,
                    return_dict=True,
                )

            step_outputs, _step_ms = measure_gpu_ms(decode_step)
            decode_forward_tokens += 1
            past_key_values = step_outputs.past_key_values
            next_token = sample_next_token(
                step_outputs.logits[:, -1, :],
                args.temperature,
                args.top_p,
            )
            generated_ids.append(next_token)

        sync_cuda()
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0
        total_latency_ms = (time.perf_counter() - latency_start) * 1000.0
        module_selector_router_time_ms = module_timer.total_ms()
        if module_selector_router_time_ms is not None:
            selector_router_time_ms += module_selector_router_time_ms
    finally:
        module_timer.close()

    generated = torch.cat(generated_ids, dim=1)
    output_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    generated_token_count = int(generated.shape[1])

    if cuda_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated()
        peak_gpu_memory_incremental = peak_gpu_memory - base_gpu_memory
    else:
        peak_gpu_memory = None
        peak_gpu_memory_incremental = None

    kv_cache_memory = tensor_bytes(past_key_values)
    vit_flops = estimate_vit_flops(vision_tower, concat_images_for_vision(images_tensor))
    projector_flops = estimate_projector_flops(projector, projector_output_token_count)
    llm_flops = estimate_llm_flops(
        model.config,
        prefill_token_count,
        decode_forward_tokens,
        generated_token_count,
        args.include_lm_head_flops,
    ) or {}
    total_flops = sum(
        value
        for value in (vit_flops, projector_flops, llm_flops.get("llm_total_flops"))
        if value is not None
    )

    return ProfileResult(
        image_preprocessing_time_ms=image_preprocessing_time_ms,
        vit_encoding_time_ms=vit_encoding_time_ms,
        projector_time_ms=projector_time_ms,
        selector_router_time_ms=selector_router_time_ms,
        selector_info=selector_info,
        llm_prefill_time_ms=llm_prefill_time_ms,
        first_token_latency_ttft_ms=first_token_latency_ttft_ms,
        decode_time_ms=decode_time_ms,
        total_latency_ms=total_latency_ms,
        peak_gpu_memory_mib=bytes_to_mib(peak_gpu_memory),
        peak_gpu_memory_incremental_mib=bytes_to_mib(peak_gpu_memory_incremental),
        kv_cache_memory_mib=bytes_to_mib(kv_cache_memory),
        flops={
            "vit_flops": vit_flops,
            "projector_flops": projector_flops,
            **llm_flops,
            "total_estimated_flops": int(total_flops) if total_flops else None,
        },
        visual_token_count=visual_token_count,
        prompt_token_count=int(input_ids.shape[1]),
        prefill_token_count=prefill_token_count,
        generated_token_count=generated_token_count,
        output_text=output_text,
    )


def print_result(result: ProfileResult) -> None:
    data = asdict(result)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Profile LLaVA inference latency, memory, KV cache, FLOPs, and visual tokens.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--method", type=str, default="llava")
    parser.add_argument("--retain-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-device", type=str, default=None)
    parser.add_argument(
        "--selector-extra",
        type=str,
        default=None,
        help="Optional JSON string for future VisionZip/FastV/Ours selector parameters.",
    )
    parser.add_argument("--include-lm-head-flops", action="store_true", help="Include LM head matmul in the FLOPs estimate.")
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    if args.device.startswith("cuda") and not cuda_available():
        raise RuntimeError("CUDA is required for GPU timing and memory metrics, but torch.cuda.is_available() is False.")

    model_name = resolve_profile_model_name(args.model_path, get_model_name_from_path(args.model_path))
    if args.conv_mode is None:
        args.conv_mode = infer_conv_mode(model_name)

    tokenizer, model, image_processor, _context_len = load_profile_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model.eval()

    for _ in range(args.warmup):
        _ = profile_once(args, tokenizer, model, image_processor)

    result = profile_once(args, tokenizer, model, image_processor)
    print_result(result)


if __name__ == "__main__":
    main()
