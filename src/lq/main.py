import torch
import numpy
import argparse

CLI_ARGS = {
    "--model": dict(
        required=True,
        choices=[
            "meta-llama/Llama-2-7b-hf",
        ],
    ),
    "--adapter_path": dict(required=True, type=str),
    "--dataset": dict(
        type=str,
        default="gsm8k",
        choices=[
            "gsm8k",
        ],
    ),
    # ["minerva_math_algebra", "humaneval", "mbpp", "EdinburghNLP/xsum"]
    "--num_fewshot": dict(type=int, default=0),
    "--num_bits_high": dict(type=int),
    "--num_bits_low": dict(type=int),
    "--group_size": dict(type=int, default=128),
    "--ratio": dict(type=float, default=0.7),
    "--method": dict(required=True, choices=["fp", "rtn", "bin", "lq", "lq_rep"]),
    "--opt": dict(action="store_true"),
    "--limit": dict(type=int),
}

def get_args():
    parser = argparse.ArgumentParser(description="LoraQuant argument parser")

    # loop over the cli args and add arguments to parser
    for arg, kwargs in CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)

    return parser.parse_args()

def main():
    args = get_args()
    print(args)



if __name__ == "__main__":
    main()