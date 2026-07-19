import argparse
import json
import math
import re
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.eval.profile_inference import (
    ProfileResult,
    bytes_to_mib,
    cuda_available,
    infer_conv_mode,
    load_profile_model,
    measure_gpu_ms,
    resolve_profile_model_name,
    sample_next_token,
    sync_cuda,
    tensor_bytes,
)
from llava.eval.profile_batch import run_profile_sample
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.efficient_vlm import disable_efficient_llama
from llava.utils import disable_torch_init


OPTIONS = ["A", "B", "C", "D", "E"]


def resolve_scienceqa_data_dir(scienceqa_root: str) -> Path:
    root = Path(scienceqa_root).expanduser()
    if (root / "problems.json").exists() and (root / "pid_splits.json").exists():
        return root
    nested = root / "data" / "scienceqa"
    if (nested / "problems.json").exists() and (nested / "pid_splits.json").exists():
        return nested
    raise FileNotFoundError(
        f"Could not find problems.json and pid_splits.json under {root} or {nested}."
    )


def resolve_image_root(scienceqa_root: str, image_root: Optional[str], split: str) -> Path:
    if image_root:
        return Path(image_root).expanduser()
    root = Path(scienceqa_root).expanduser()
    if (root / split).exists():
        return root
    repo_root = root.parent.parent
    if (repo_root / split).exists():
        return repo_root
    return root


def get_context(problem: Dict[str, Any], use_caption: bool = False) -> str:
    hint = problem.get("hint") or ""
    caption = (problem.get("caption") or "") if use_caption else ""
    context = " ".join([hint, caption]).strip()
    return context or "N/A"


def get_choices(problem: Dict[str, Any]) -> str:
    return " ".join(
        f"({OPTIONS[i]}) {choice}" for i, choice in enumerate(problem["choices"])
    )


def build_sqa_prompt(problem: Dict[str, Any], prompt_format: str) -> str:
    input_format, _output_format = prompt_format.split("-")
    question = problem["question"]
    context = get_context(problem, use_caption=False)
    choices = get_choices(problem)
    lecture = (problem.get("lecture") or "").replace("\n", "\\n")
    solution = (problem.get("solution") or "").replace("\n", "\\n")

    if input_format == "CQM":
        prompt = f"Context: {context}\nQuestion: {question}\nOptions: {choices}\n"
    elif input_format == "QCM":
        prompt = f"Question: {question}\nContext: {context}\nOptions: {choices}\n"
    elif input_format == "QCML":
        prompt = f"Question: {question}\nContext: {context}\nOptions: {choices}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        prompt = f"Question: {question}\nContext: {context}\nOptions: {choices}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        prompt = f"Question: {question}\nContext: {context}\nOptions: {choices}\nBECAUSE: {lecture} {solution}\n"
    elif input_format == "QCLM":
        prompt = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choices}\n"
    elif input_format == "QCEM":
        prompt = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choices}\n"
    elif input_format == "QCLEM":
        prompt = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choices}\n"
    else:
        raise ValueError(f"Unsupported ScienceQA prompt input format: {input_format}")

    prompt = prompt.replace("  ", " ").strip()
    if prompt.endswith("BECAUSE:"):
        prompt = prompt.replace("BECAUSE:", "").strip()
    return prompt


def parse_answer(pred_text: str, choices: List[str]) -> Tuple[str, int]:
    pred_text = (pred_text or "").strip()
    valid = OPTIONS[: len(choices)]
    if pred_text in valid:
        answer = pred_text
    elif pred_text and pred_text[0] in valid and re.match(r"^[A-Z](?:[\.\)]|\s|$)", pred_text):
        answer = pred_text[0]
    else:
        patterns = [
            r"^\(([A-Z])\)",
            r"The answer is ([A-Z])\.",
            r"answer is ([A-Z])",
            r"option ([A-Z])",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, pred_text, flags=re.IGNORECASE))
        normalized = [m.upper() for m in matches if m.upper() in valid]
        answer = normalized[0] if len(set(normalized)) == 1 else "FAILED"
    return answer, OPTIONS.index(answer) if answer in valid else -1


def split_list(items: List[Any], n_chunks: int) -> List[List[Any]]:
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: List[Any], n_chunks: int, chunk_idx: int) -> List[Any]:
    return split_list(items, n_chunks)[chunk_idx]


def iter_samples(args) -> Iterator[Dict[str, Any]]:
    data_dir = resolve_scienceqa_data_dir(args.scienceqa_root)
    image_root = resolve_image_root(args.scienceqa_root, args.image_root, args.split)
    split_indices = json.load(open(data_dir / "pid_splits.json", "r"))[args.split]
    problems = json.load(open(data_dir / "problems.json", "r"))

    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive.")
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("--chunk-idx must be in [0, num_chunks).")
    split_indices = get_chunk(split_indices, args.num_chunks, args.chunk_idx)
    if args.limit is not None:
        split_indices = split_indices[: args.limit]

    for prob_id in split_indices:
        problem = problems[prob_id]
        if args.multimodal_only and problem.get("image") is None:
            continue

        query = build_sqa_prompt(problem, args.prompt_format)
        if args.single_pred_prompt:
            query = query + "\nAnswer with the option's letter from the given choices directly."

        image_file = None
        prompt_for_eval = query
        if problem.get("image") is not None:
            image_file = image_root / args.split / prob_id / problem["image"]
            prompt_for_eval = "<image>\n" + query

        yield {
            "id": prob_id,
            "image": str(image_file) if image_file is not None else None,
            "question": query,
            "answer": OPTIONS[problem["answer"]],
            "answer_idx": problem["answer"],
            "choices": problem["choices"],
            "category": problem.get("category"),
            "subject": problem.get("subject"),
            "topic": problem.get("topic"),
            "split": args.split,
            "prompt_for_eval": prompt_for_eval,
            "is_multimodal": image_file is not None,
        }


@contextmanager
def text_sample_args(args, sample: Dict[str, Any]):
    old_query = getattr(args, "query", None)
    args.query = sample["question"]
    try:
        yield args
    finally:
        args.query = old_query


@torch.inference_mode()
def profile_text_once(args, tokenizer_or_processor, model) -> ProfileResult:
    disable_efficient_llama(model)
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    model_device = getattr(model, "device", torch.device(args.device))
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], args.query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model_device)

    if cuda_available():
        torch.cuda.empty_cache()
        base_gpu_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        base_gpu_memory = None

    latency_start = time.perf_counter()

    def prefill():
        return model(input_ids=input_ids, use_cache=True, return_dict=True)

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
    prefill_token_count = int(input_ids.shape[1])
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
    generated = torch.cat(generated_ids, dim=1)
    output_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    if cuda_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated()
        peak_gpu_memory_incremental = peak_gpu_memory - base_gpu_memory
    else:
        peak_gpu_memory = None
        peak_gpu_memory_incremental = None

    return ProfileResult(
        image_preprocessing_time_ms=0.0,
        vit_encoding_time_ms=None,
        projector_time_ms=None,
        selector_router_time_ms=None,
        selector_info=None,
        llm_prefill_time_ms=llm_prefill_time_ms,
        first_token_latency_ttft_ms=first_token_latency_ttft_ms,
        decode_time_ms=decode_time_ms,
        total_latency_ms=total_latency_ms,
        peak_gpu_memory_mib=bytes_to_mib(peak_gpu_memory),
        peak_gpu_memory_incremental_mib=bytes_to_mib(peak_gpu_memory_incremental),
        kv_cache_memory_mib=bytes_to_mib(tensor_bytes(past_key_values)),
        flops={
            "vit_flops": None,
            "projector_flops": None,
            "llm_prefill_flops": None,
            "llm_decode_flops": None,
            "llm_total_flops": None,
            "generated_tokens_for_flops": int(generated.shape[1]),
            "total_estimated_flops": None,
        },
        visual_token_count=0,
        prompt_token_count=int(input_ids.shape[1]),
        prefill_token_count=prefill_token_count,
        generated_token_count=int(generated.shape[1]),
        output_text=output_text,
    )


def profile_sample(args, sample, tokenizer, model, image_processor):
    if sample["is_multimodal"]:
        return run_profile_sample(args, sample, tokenizer, model, image_processor)
    with text_sample_args(args, sample):
        return profile_text_once(args, tokenizer, model)


def mean_metric(rows: List[Dict[str, Any]], metric: str) -> Optional[float]:
    values = []
    for row in rows:
        result = row.get("profile_result") or {}
        value = result.get(metric)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def build_summary(rows: List[Dict[str, Any]], errors: int) -> Dict[str, Any]:
    evaluated = [row for row in rows if row.get("pred_idx") is not None]
    correct = [row for row in evaluated if row.get("correct")]
    multimodal = [row for row in evaluated if row.get("is_multimodal")]
    multimodal_correct = [row for row in multimodal if row.get("correct")]
    timing_metrics = [
        "image_preprocessing_time_ms",
        "vit_encoding_time_ms",
        "projector_time_ms",
        "selector_router_time_ms",
        "llm_prefill_time_ms",
        "first_token_latency_ttft_ms",
        "decode_time_ms",
        "total_latency_ms",
    ]
    return {
        "count": len(evaluated),
        "correct": len(correct),
        "accuracy": len(correct) / len(evaluated) * 100 if evaluated else None,
        "multimodal_count": len(multimodal),
        "multimodal_correct": len(multimodal_correct),
        "multimodal_accuracy": (
            len(multimodal_correct) / len(multimodal) * 100 if multimodal else None
        ),
        "errors": errors,
        "average_profile": {metric: mean_metric(rows, metric) for metric in timing_metrics},
    }


def make_answer_id() -> str:
    return uuid.uuid4().hex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile LLaVA on ScienceQA and report stage latency plus accuracy."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--scienceqa-root", type=str, default="/root/autodl-tmp/ScienceQA")
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt-format", type=str, default="CQM-A")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--summary-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-n-samples", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--multimodal-only", action="store_true")
    parser.add_argument("--method", type=str, default="llava")
    parser.add_argument("--retain-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-device", type=str, default=None)
    parser.add_argument("--selector-extra", type=str, default=None)
    parser.add_argument("--include-lm-head-flops", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    if args.device.startswith("cuda") and not cuda_available():
        raise RuntimeError(
            "CUDA is required for GPU timing and memory metrics, but torch.cuda.is_available() is False."
        )
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    samples = list(iter_samples(args))
    model_name = resolve_profile_model_name(
        args.model_path, get_model_name_from_path(args.model_path)
    )
    if args.conv_mode is None:
        args.conv_mode = infer_conv_mode(model_name)
    selector_extra = json.loads(args.selector_extra) if args.selector_extra else {}

    tokenizer, model, image_processor, _context_len = load_profile_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
        method=args.method,
        retain_tokens=args.retain_tokens,
        selector_extra=selector_extra,
    )
    model.eval()

    for sample in samples[: args.warmup_n_samples]:
        try:
            _ = profile_sample(args, sample, tokenizer, model, image_processor)
        except Exception as exc:
            print(f"Warmup sample {sample.get('id', '<unknown>')} failed: {exc}", file=sys.stderr)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path = (
        Path(args.answers_file)
        if args.answers_file
        else output_path.with_suffix(".answers.jsonl")
    )
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_file)
        if args.summary_file
        else output_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = 0
    with open(output_path, "w", encoding="utf-8") as profile_f, open(
        answers_path, "w", encoding="utf-8"
    ) as answers_f:
        for sample in samples:
            try:
                result = profile_sample(args, sample, tokenizer, model, image_processor)
                pred_text = result.output_text
                parsed_answer, pred_idx = parse_answer(pred_text, sample["choices"])
                correct = pred_idx == sample["answer_idx"]
                error = None
                profile_result = asdict(result)
            except Exception as exc:
                errors += 1
                pred_text = "FAILED"
                parsed_answer = "FAILED"
                pred_idx = -1
                correct = False
                error = repr(exc)
                profile_result = None

            answer_row = {
                "question_id": sample["id"],
                "prompt": sample["prompt_for_eval"],
                "text": pred_text,
                "answer_id": make_answer_id(),
                "model_id": model_name,
                "metadata": {},
            }
            answers_f.write(json.dumps(answer_row, ensure_ascii=False) + "\n")
            answers_f.flush()

            row = {
                **sample,
                "method": args.method,
                "retain_tokens": args.retain_tokens,
                "seed": args.seed,
                "selector_extra": args.selector_extra,
                "prediction": pred_text,
                "parsed_answer": parsed_answer,
                "pred_idx": pred_idx,
                "correct": correct,
                "profile_result": profile_result,
                "error": error,
            }
            profile_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            profile_f.flush()
            rows.append(row)

    summary = build_summary(rows, errors)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    acc = summary["accuracy"]
    img_acc = summary["multimodal_accuracy"]
    if acc is None:
        print("Total: 0, Correct: 0, Accuracy: N/A")
    else:
        print(
            f"Total: {summary['count']}, Correct: {summary['correct']}, "
            f"Accuracy: {acc:.2f}%"
        )
    if img_acc is not None:
        print(
            f"IMG-Total: {summary['multimodal_count']}, "
            f"IMG-Correct: {summary['multimodal_correct']}, IMG-Accuracy: {img_acc:.2f}%"
        )
    print(f"Wrote profile rows to {output_path}")
    print(f"Wrote compatible answers to {answers_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
