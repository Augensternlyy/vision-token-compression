import argparse
import csv
import json
import math
import statistics
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SUMMARY_METRICS = [
    "image_preprocessing_time_ms",
    "vit_encoding_time_ms",
    "projector_time_ms",
    "selector_router_time_ms",
    "llm_prefill_time_ms",
    "first_token_latency_ttft_ms",
    "decode_time_ms",
    "decode_throughput_tokens_per_s",
    "total_latency_ms",
    "peak_gpu_memory_incremental_mib",
    "kv_cache_memory_mib",
    "visual_token_count",
    "prompt_token_count",
    "prefill_token_count",
    "generated_token_count",
    "flops.llm_prefill_flops",
    "flops.llm_decode_flops",
    "flops.total_estimated_flops",
    "llm_prefill_tflops",
    "total_estimated_tflops",
    "selector.original_visual_tokens",
    "selector.retained_visual_tokens",
]

SUMMARY_COLUMNS = [
    "method",
    "retain_tokens",
    "count",
    *[f"{metric}_mean" for metric in SUMMARY_METRICS],
    "prefill_speedup_vs_baseline",
    "ttft_speedup_vs_baseline",
    "memory_ratio_vs_baseline",
    "flops_ratio_vs_baseline",
]

REQUIRED_SAMPLE_FIELDS = ("id", "image", "question", "answer", "category")


def parse_int_list(values: Sequence[str]) -> List[int]:
    tokens: List[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            tokens.append(int(part))
    if not tokens:
        raise ValueError("At least one retain token value is required.")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLaVA profiling for several compression token counts in one model load, "
            "then export a compact summary CSV."
        )
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional raw per-sample profile output, including failures.",
    )
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
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated profiling passes per sample/config.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["first_n", "random", "spatial_uniform"],
        help="Compression methods to profile. Example: --methods first_n random spatial_uniform",
    )
    parser.add_argument(
        "--retain-tokens-list",
        nargs="+",
        default=["32", "64", "128", "256"],
        help="Retained visual token counts. Supports spaces or commas, e.g. 32 64 128 256 or 32,64,128,256.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the uncompressed llava baseline row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-device", type=str, default=None)
    parser.add_argument(
        "--selector-extra",
        type=str,
        default=None,
        help="Optional JSON string for future VisionZip/FastV/Ours selector parameters.",
    )
    parser.add_argument(
        "--include-lm-head-flops",
        action="store_true",
        help="Include LM head matmul in the FLOPs estimate.",
    )
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def load_samples(path: str) -> List[Dict[str, Any]]:
    samples = list(iter_jsonl(path))
    for idx, sample in enumerate(samples, start=1):
        missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"Sample {idx} is missing required field(s): {', '.join(missing)}")
    return samples


@contextmanager
def sample_args(args: argparse.Namespace, sample: Dict[str, Any]) -> Iterator[argparse.Namespace]:
    old_image_file = getattr(args, "image_file", None)
    old_query = getattr(args, "query", None)
    args.image_file = sample["image"]
    args.query = sample["question"]
    try:
        yield args
    finally:
        args.image_file = old_image_file
        args.query = old_query


def run_profile_sample(
    profile_once: Callable,
    args: argparse.Namespace,
    sample: Dict[str, Any],
    tokenizer,
    model,
    image_processor,
):
    with sample_args(args, sample):
        return profile_once(args, tokenizer, model, image_processor)


def as_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    numerator = as_number(numerator)
    denominator = as_number(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def config_grid(methods: Iterable[str], retain_tokens: Iterable[int], include_baseline: bool) -> List[Tuple[str, Optional[int]]]:
    configs: List[Tuple[str, Optional[int]]] = []
    if include_baseline:
        configs.append(("llava", None))
    for method in methods:
        if method.lower() in {"llava", "none", "vanilla"}:
            configs.append((method, None))
            continue
        for token_count in retain_tokens:
            configs.append((method, token_count))
    return configs


def result_to_row(
    sample: Dict[str, Any],
    method: str,
    retain_tokens: Optional[int],
    repeat: int,
    seed: int,
    selector_extra: Optional[str],
    result,
) -> Dict[str, Any]:
    profile = asdict(result)
    flops = profile.get("flops") or {}
    selector_info = profile.get("selector_info") or {}
    generated = as_number(profile.get("generated_token_count")) or 0.0
    decode_time = as_number(profile.get("decode_time_ms"))
    decode_tokens = max(generated - 1.0, 0.0)

    row: Dict[str, Any] = {
        **sample,
        "method": method,
        "retain_tokens": retain_tokens,
        "repeat": repeat,
        "seed": seed,
        "selector_extra": selector_extra,
        "error": None,
    }
    for key, value in profile.items():
        if key in {"flops", "selector_info", "output_text"}:
            continue
        row[key] = value
    for key, value in flops.items():
        row[f"flops.{key}"] = value
    row["selector.original_visual_tokens"] = selector_info.get("original_visual_tokens")
    row["selector.retained_visual_tokens"] = selector_info.get("retained_visual_tokens")
    row["decode_throughput_tokens_per_s"] = (
        decode_tokens / (decode_time / 1000.0)
        if decode_tokens > 0 and decode_time is not None and decode_time > 0
        else None
    )
    prefill_flops = as_number(row.get("flops.llm_prefill_flops"))
    total_flops = as_number(row.get("flops.total_estimated_flops"))
    row["llm_prefill_tflops"] = prefill_flops / 1e12 if prefill_flops is not None else None
    row["total_estimated_tflops"] = total_flops / 1e12 if total_flops is not None else None
    return row


def error_row(
    sample: Dict[str, Any],
    method: str,
    retain_tokens: Optional[int],
    repeat: int,
    seed: int,
    selector_extra: Optional[str],
    error: Exception,
) -> Dict[str, Any]:
    return {
        **sample,
        "method": method,
        "retain_tokens": retain_tokens,
        "repeat": repeat,
        "seed": seed,
        "selector_extra": selector_extra,
        "error": repr(error),
    }


def write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def mean_metric(rows: Sequence[Dict[str, Any]], metric: str) -> Optional[float]:
    values = [as_number(row.get(metric)) for row in rows]
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (str(row.get("method")), "" if row.get("retain_tokens") is None else str(row.get("retain_tokens")))
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (method, retain_tokens), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], as_number(item[0][1]) or -1)):
        summary: Dict[str, Any] = {
            "method": method,
            "retain_tokens": retain_tokens,
            "count": len(group_rows),
        }
        for metric in SUMMARY_METRICS:
            summary[f"{metric}_mean"] = mean_metric(group_rows, metric)
        summary_rows.append(summary)

    baseline = next((row for row in summary_rows if row["method"] == "llava"), None)
    baseline_prefill = baseline.get("llm_prefill_time_ms_mean") if baseline else None
    baseline_ttft = baseline.get("first_token_latency_ttft_ms_mean") if baseline else None
    baseline_memory = baseline.get("peak_gpu_memory_incremental_mib_mean") if baseline else None
    baseline_flops = baseline.get("flops.total_estimated_flops_mean") if baseline else None

    for row in summary_rows:
        row["prefill_speedup_vs_baseline"] = safe_ratio(baseline_prefill, row.get("llm_prefill_time_ms_mean"))
        row["ttft_speedup_vs_baseline"] = safe_ratio(baseline_ttft, row.get("first_token_latency_ttft_ms_mean"))
        row["memory_ratio_vs_baseline"] = safe_ratio(row.get("peak_gpu_memory_incremental_mib_mean"), baseline_memory)
        row["flops_ratio_vs_baseline"] = safe_ratio(row.get("flops.total_estimated_flops_mean"), baseline_flops)
    return summary_rows


def write_summary_csv(rows: Sequence[Dict[str, Any]], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def main() -> None:
    args = parse_args()

    from llava.eval.profile_inference import (
        cuda_available,
        infer_conv_mode,
        load_profile_model,
        profile_once,
        resolve_profile_model_name,
    )
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    disable_torch_init()

    if args.device.startswith("cuda") and not cuda_available():
        raise RuntimeError("CUDA is required for GPU timing and memory metrics, but torch.cuda.is_available() is False.")
    if args.warmup_n_samples < 0:
        raise ValueError("--warmup-n-samples must be non-negative.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    retain_tokens = parse_int_list(args.retain_tokens_list)
    configs = config_grid(args.methods, retain_tokens, include_baseline=not args.no_baseline)
    samples = load_samples(args.question_file)
    eval_samples = samples[: args.limit] if args.limit is not None else samples

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

    for sample in samples[: args.warmup_n_samples]:
        try:
            args.method = configs[0][0]
            args.retain_tokens = configs[0][1]
            _ = run_profile_sample(profile_once, args, sample, tokenizer, model, image_processor)
        except Exception as exc:
            print(f"Warmup sample {sample.get('id', '<unknown>')} failed: {exc}", file=sys.stderr)

    raw_handle = None
    if args.output_jsonl:
        raw_path = Path(args.output_jsonl)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = raw_path.open("w", encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    try:
        total_runs = args.repeats * len(eval_samples) * len(configs)
        run_idx = 0
        for repeat in range(args.repeats):
            for sample in eval_samples:
                for method, token_count in configs:
                    args.method = method
                    args.retain_tokens = token_count
                    run_idx += 1
                    print(
                        f"[{run_idx}/{total_runs}] repeat={repeat} method={method} retain_tokens={token_count} sample={sample.get('id', '<unknown>')}",
                        file=sys.stderr,
                    )
                    try:
                        result = run_profile_sample(profile_once, args, sample, tokenizer, model, image_processor)
                        row = result_to_row(sample, method, token_count, repeat, args.seed, args.selector_extra, result)
                    except Exception as exc:
                        row = error_row(sample, method, token_count, repeat, args.seed, args.selector_extra, exc)
                        print(f"Profile failed: {row['error']}", file=sys.stderr)
                    rows.append(row)
                    if raw_handle is not None:
                        write_jsonl_row(raw_handle, row)
    finally:
        if raw_handle is not None:
            raw_handle.close()

    summary_rows = summarize(rows)
    write_summary_csv(summary_rows, args.output_csv)


if __name__ == "__main__":
    main()
