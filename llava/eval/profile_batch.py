import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from llava.eval.profile_inference import (
    cuda_available,
    infer_conv_mode,
    load_profile_model,
    profile_once,
    resolve_profile_model_name,
)
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init


REQUIRED_SAMPLE_FIELDS = ("id", "image", "question", "answer", "category")


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def load_samples(path: str) -> List[Dict]:
    samples = list(iter_jsonl(path))
    for idx, sample in enumerate(samples, start=1):
        missing = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"Sample {idx} is missing required field(s): {', '.join(missing)}")
    return samples


@contextmanager
def sample_args(args, sample: Dict):
    old_image_file = getattr(args, "image_file", None)
    old_query = getattr(args, "query", None)
    args.image_file = sample["image"]
    args.query = sample["question"]
    try:
        yield args
    finally:
        args.image_file = old_image_file
        args.query = old_query


def run_profile_sample(args, sample: Dict, tokenizer, model, image_processor):
    with sample_args(args, sample):
        return profile_once(args, tokenizer, model, image_processor)   

def write_result(f, sample: Dict, method: str, args, profile_result=None, error: Optional[str] = None) -> None:
    row = {
        **sample,
        "method": method,
        "retain_tokens": args.retain_tokens,
        "seed": args.seed,
        "selector_extra": args.selector_extra,
        "profile_result": asdict(profile_result) if profile_result is not None else None,
        "error": error,
    }
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch profile LLaVA inference over a JSONL question file."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
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
    parser.add_argument(
        "--include-lm-head-flops",
        action="store_true",
        help="Include LM head matmul in the FLOPs estimate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    if args.device.startswith("cuda") and not cuda_available():
        raise RuntimeError("CUDA is required for GPU timing and memory metrics, but torch.cuda.is_available() is False.")
    if args.warmup_n_samples < 0:
        raise ValueError("--warmup-n-samples must be non-negative.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    samples = load_samples(args.question_file)
    eval_samples = samples[: args.limit] if args.limit is not None else samples

    model_name = resolve_profile_model_name(args.model_path, get_model_name_from_path(args.model_path))
    if args.conv_mode is None:
        args.conv_mode = infer_conv_mode(model_name)
    method = args.method
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
            _ = run_profile_sample(args, sample, tokenizer, model, image_processor)
        except Exception as exc:
            print(f"Warmup sample {sample.get('id', '<unknown>')} failed: {exc}", file=sys.stderr)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in eval_samples:
            try:
                result = run_profile_sample(args, sample, tokenizer, model, image_processor)
                write_result(f, sample, method, args, profile_result=result, error=None)
            except Exception as exc:
                write_result(f, sample, method, args, profile_result=None, error=repr(exc))


if __name__ == "__main__":
    main()
