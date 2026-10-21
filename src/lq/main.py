import torch
import numpy
import argparse
import os
from pathlib import Path

CLI_ARGS = {
    "--model": dict(
        required=True,
        choices=[
            "meta-llama/Llama-2-7b-hf",
        ],
    ),
    "--method": dict(required=True, choices=["fp", "rtn", "bin", "lq", "lq_rep"]),
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
    "--num_bits_high": dict(type=int),
    "--num_bits_low": dict(type=int),
    "--group_size": dict(type=int, default=128),
    "--ratio": dict(type=float, default=0.7),
    "--opt": dict(action="store_true"),
    "--limit": dict(type=int),
    "--results_dir": dict(type=str, default="outputs"),
}


# GET COMMAND LINE ARGUMENTS
def get_args():
    parser = argparse.ArgumentParser(description="LoraQuant argument parser")

    # loop over the cli args and add arguments to parser
    for arg, kwargs in CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)

    return parser.parse_args()


def validate_args(args):
    if args.model not in ["meta-llama/Llama-2-7b-hf"]:
        raise ValueError(
            f"Invalid model: {args.model}. Must be one of: ['meta-llama/Llama-2-7b-hf']"
        )


# CONVERT ARGUMENTS INTO PATH FOR RESULTS
def format_results_path(args) -> Path:
    # shorten the model name
    model_name = args.model.split("/")[-1].lower()

    path = Path(args.results_dir) / model_name / args.dataset / args.method

    return path


def main():
    args = get_args()
    path = format_results_path(args)
    # os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    main()
