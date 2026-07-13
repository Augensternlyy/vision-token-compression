import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROFILE_FIELDS = [
    "image_preprocessing_time_ms",
    "vit_encoding_time_ms",
    "projector_time_ms",
    "selector_router_time_ms",
    "llm_prefill_time_ms",
    "first_token_latency_ttft_ms",
    "decode_time_ms",
    "total_latency_ms",
    "peak_gpu_memory_mib",
    "peak_gpu_memory_incremental_mib",
    "kv_cache_memory_mib",
    "visual_token_count",
    "prompt_token_count",
    "prefill_token_count",
    "generated_token_count",
]

FLOP_FIELDS = [
    "vit_flops",
    "projector_flops",
    "llm_prefill_flops",
    "llm_decode_flops",
    "llm_total_flops",
    "total_estimated_flops",
]

SELECTOR_FIELDS = [
    "selector.original_visual_tokens",
    "selector.retained_visual_tokens",
]

DERIVED_FIELDS = [
    "decode_ms_per_token",
    "prefill_tflops",
    "total_tflops",
]

SUMMARY_METRICS = PROFILE_FIELDS + [f"flops.{name}" for name in FLOP_FIELDS] + SELECTOR_FIELDS + DERIVED_FIELDS

MARKDOWN_COLUMNS = [
    "method",
    "retain_tokens",
    "count",
    "visual_token_count_mean",
    "prefill_token_count_mean",
    "llm_prefill_time_ms_mean",
    "first_token_latency_ttft_ms_mean",
    "decode_ms_per_token_mean",
    "total_latency_ms_mean",
    "kv_cache_memory_mib_mean",
    "peak_gpu_memory_incremental_mib_mean",
    "prefill_tflops_mean",
    "total_tflops_mean",
    "prefill_speedup",
    "ttft_speedup",
    "total_speedup",
]


try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    pd = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze profile_batch.py JSONL outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more profile JSONL files.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path for summary CSV output.")
    parser.add_argument("--output-md", type=str, default=None, help="Optional path for markdown summary output.")
    return parser.parse_args()


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


def load_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for input_path in paths:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc

                profile = record.get("profile_result")
                if record.get("error") or profile is None:
                    continue

                flat: Dict[str, Any] = {
                    "input_file": str(path),
                    "method": record.get("method"),
                    "retain_tokens": record.get("retain_tokens"),
                }
                for field in PROFILE_FIELDS:
                    flat[field] = as_number(profile.get(field))

                flops = profile.get("flops") or {}
                for field in FLOP_FIELDS:
                    flat[f"flops.{field}"] = as_number(flops.get(field))

                selector_info = profile.get("selector_info") or {}
                flat["selector.original_visual_tokens"] = as_number(
                    selector_info.get("original_visual_tokens")
                )
                flat["selector.retained_visual_tokens"] = as_number(
                    selector_info.get("retained_visual_tokens")
                )

                generated = as_number(profile.get("generated_token_count")) or 0.0
                decode_time = as_number(profile.get("decode_time_ms"))
                flat["decode_ms_per_token"] = (
                    decode_time / max(generated - 1.0, 1.0) if decode_time is not None else None
                )

                prefill_flops = flat.get("flops.llm_prefill_flops")
                total_flops = flat.get("flops.total_estimated_flops")
                flat["prefill_tflops"] = prefill_flops / 1e12 if prefill_flops is not None else None
                flat["total_tflops"] = total_flops / 1e12 if total_flops is not None else None
                rows.append(flat)
    return rows


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return clean[int(pos)]
    weight = pos - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def normalize_retain_key(value: Any) -> Any:
    return "" if value is None else value


def add_speedups(summary_rows: List[Dict[str, Any]]) -> None:
    baseline_rows = [row for row in summary_rows if str(row.get("method")) == "llava"]
    if not baseline_rows:
        for row in summary_rows:
            row["prefill_speedup"] = None
            row["ttft_speedup"] = None
            row["total_speedup"] = None
        return

    baseline = next((row for row in baseline_rows if row.get("retain_tokens") in (None, "")), baseline_rows[0])
    baseline_prefill = baseline.get("llm_prefill_time_ms_mean")
    baseline_ttft = baseline.get("first_token_latency_ttft_ms_mean")
    baseline_total = baseline.get("total_latency_ms_mean")

    for row in summary_rows:
        row["prefill_speedup"] = safe_ratio(baseline_prefill, row.get("llm_prefill_time_ms_mean"))
        row["ttft_speedup"] = safe_ratio(baseline_ttft, row.get("first_token_latency_ttft_ms_mean"))
        row["total_speedup"] = safe_ratio(baseline_total, row.get("total_latency_ms_mean"))


def safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    numerator = as_number(numerator)
    denominator = as_number(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def summarize_with_pandas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(rows)
    df["retain_tokens"] = df["retain_tokens"].where(df["retain_tokens"].notna(), None)
    grouped = df.groupby(["method", "retain_tokens"], dropna=False, sort=True)
    summary = grouped.size().reset_index(name="count")

    for metric in SUMMARY_METRICS:
        if metric not in df.columns:
            continue
        stats = grouped[metric].agg(
            **{
                f"{metric}_mean": "mean",
                f"{metric}_median": "median",
                f"{metric}_std": "std",
                f"{metric}_p90": lambda x: x.quantile(0.9),
            }
        ).reset_index()
        summary = summary.merge(stats, on=["method", "retain_tokens"], how="left")

    summary = summary.where(summary.notna(), None)
    output = summary.to_dict(orient="records")
    for row in output:
        retain_tokens = row.get("retain_tokens")
        if retain_tokens is None or (isinstance(retain_tokens, float) and math.isnan(retain_tokens)):
            row["retain_tokens"] = ""
        elif isinstance(retain_tokens, float) and retain_tokens.is_integer():
            row["retain_tokens"] = int(retain_tokens)
    add_speedups(output)
    return output


def summarize_without_pandas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("method"), normalize_retain_key(row.get("retain_tokens")))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (method, retain_tokens), group_rows in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        summary: Dict[str, Any] = {
            "method": method,
            "retain_tokens": retain_tokens,
            "count": len(group_rows),
        }
        for metric in SUMMARY_METRICS:
            values = [as_number(row.get(metric)) for row in group_rows]
            clean = [v for v in values if v is not None]
            summary[f"{metric}_mean"] = statistics.fmean(clean) if clean else None
            summary[f"{metric}_median"] = statistics.median(clean) if clean else None
            summary[f"{metric}_std"] = statistics.stdev(clean) if len(clean) > 1 else None
            summary[f"{metric}_p90"] = percentile(clean, 0.9)
        summary_rows.append(summary)

    add_speedups(summary_rows)
    return summary_rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if pd is not None:
        return summarize_with_pandas(rows)
    return summarize_without_pandas(rows)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        abs_value = abs(value)
        if abs_value >= 1000:
            return f"{value:.2f}"
        if abs_value >= 10:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def build_warnings(rows: List[Dict[str, Any]]) -> List[str]:
    warnings = []
    no_prune_methods = {"llava", "none", "vanilla"}
    for row in rows:
        method = str(row.get("method") or "")
        if method.lower() in no_prune_methods:
            continue
        original = as_number(row.get("selector.original_visual_tokens_mean"))
        retained = as_number(row.get("selector.retained_visual_tokens_mean"))
        visual = as_number(row.get("visual_token_count_mean"))
        if original is None:
            original = visual
        if retained is None:
            retained = visual
        if original is not None and retained is not None and retained >= original:
            warnings.append(
                f"- `{method}` retain_tokens={format_value(row.get('retain_tokens')) or 'null'} "
                f"did not prune visual tokens ({format_value(retained)}/{format_value(original)} retained)."
            )
    return warnings


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(MARKDOWN_COLUMNS) + " |"
    divider = "| " + " | ".join(["---"] * len(MARKDOWN_COLUMNS)) + " |"
    lines = []
    warnings = build_warnings(rows)
    if warnings:
        lines.extend(["**Warnings**", "", *warnings, ""])
    lines.extend([header, divider])
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col)) for col in MARKDOWN_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def csv_columns(rows: List[Dict[str, Any]]) -> List[str]:
    base = ["method", "retain_tokens", "count"]
    stat_cols: List[str] = []
    for metric in SUMMARY_METRICS:
        stat_cols.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std", f"{metric}_p90"])
    tail = ["prefill_speedup", "ttft_speedup", "total_speedup"]
    columns = base + stat_cols + tail
    extras = sorted({key for row in rows for key in row.keys()} - set(columns))
    return columns + extras


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = csv_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.inputs)
    summary_rows = summarize(rows)
    md = markdown_table(summary_rows)

    if args.output_csv:
        write_csv(summary_rows, args.output_csv)
    if args.output_md:
        write_text(md, args.output_md)
    if not args.output_md:
        print(md, end="")


if __name__ == "__main__":
    main()
