from pathlib import Path
from typing import Any
import argparse
import os

CLI_ARGS: dict[str, Any] = {
    "--model": dict(
        required=True,
        choices=[
            "meta-llama/Llama-2-7b-hf",
        ],
    ),
    "--method": dict(required=True, choices=["fp", "rtn", "bin", "lq"]),
    "--dataset": dict(
        type=str,
        default="gsm8k",
        choices=[
            "gsm8k",
        ],
    ),
    "--adapter_path": dict(required=True, type=str),
    # ["minerva_math_algebra", "humaneval", "mbpp", "EdinburghNLP/xsum"]
    # "--num_fewshot": dict(type=int, default=0),
    "--group_size": dict(type=int, default=128),
    "--high_bits": dict(type=int),
    "--low_bits": dict(type=int),
    "--ratio": dict(type=float, default=0.7),
    "--opt": dict(action="store_true"),
    "--opt_steps": dict(type=int, default=100),
    "--lr": dict(type=float, default=1e-3),
    "--svd": dict(choices=["full", "qr"], default="full"),
    "--limit": dict(type=int),
    "--res_path": dict(type=str, default="outputs"),
}


# parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description="LoraQuant argument parser")

    # loop over the cli args and add arguments to parser
    for arg, kwargs in CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)

    args = parser.parse_args()
    validate_args(parser, args)

    return args

# validation logic for command-line arguments
def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not os.path.isdir(args.adapter_path):
        parser.error(f"--adapter_path {args.adapter_path} is not a directory")

    if args.method in ["rtn", "lq"] and args.high_bits is None:
        parser.error(f"--method {args.method} requires --high_bits")

    if args.high_bits is not None and args.high_bits < 1:
        parser.error(f"--high_bits must be >= 1, got {args.high_bits}")

    if args.group_size <= 0:
        parser.error(f"--group_size must be > 0, got {args.group_size}")

    if args.method == "lq" and not 0 < args.ratio <= 1:
        parser.error(f"--ratio must be in (0, 1], got {args.ratio}")

    if args.opt:
        if args.method != "lq":
            parser.error("--opt only applies to --method lq")
        if args.opt_steps <= 0:
            parser.error(f"--opt_steps must be > 0, got {args.opt_steps}")
        if args.lr <= 0:
            parser.error(f"--lr must be > 0, got {args.lr}")

    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be > 0, got {args.limit}")

    if args.svd != "full" and args.method != "lq":
        parser.error(f"--svd {args.svd} only applies to --method lq")


# construct the results path
def format_results_path(args: argparse.Namespace) -> Path:
    # shorten the model name
    model_name = args.model.split("/")[-1].lower()

    # if the method quantizes the adapter
    quantized = args.method != "fp"

    # lq and bin always quantize the low part to 1 bit, so --low_bits is unused there
    uses_low_bits = args.method not in ["lq", "bin"]

    # STE optimization only applies to lq
    opt = args.method == "lq" and args.opt

    options = {
        "hi": args.high_bits if args.method in ["rtn", "lq"] else None,
        "lo": args.low_bits if uses_low_bits else None,
        "grp": args.group_size if quantized else None,
        "rho": args.ratio if args.method == "lq" else None,  # ratio is only used by lq
        "opt": args.opt_steps if opt else None,
        "lr": args.lr if opt else None,
        "svd": args.svd if args.method == "lq" and args.svd != "full" else None,
        "lim": args.limit,
    }

    filename_parts = [args.method] + [
        f"{opt}{val}" for opt, val in options.items() if val is not None
    ]

    filename = "_".join(filename_parts) + ".json"

    return Path(args.res_path) / model_name / args.dataset / args.method / filename
