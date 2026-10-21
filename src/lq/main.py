from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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


# CONVERT ARGUMENTS INTO PATH FOR RESULTS
def format_results_path(args) -> Path:
    # shorten the model name
    model_name = args.model.split("/")[-1].lower()

    path = Path(args.results_dir) / model_name / args.dataset / args.method

    return path


def main():
    # set device for torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # grab CLAs and output path
    args = get_args()
    path = format_results_path(args)
    # os.makedirs(path, exist_ok=True)

    hf_access_token = os.environ.get("HF_TOKEN", default=None)

    compute_data_type = getattr(torch, "float16")

    # MODEL QUANTIZATION CONFIG
    bits_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_data_type,
        bnb_4bit_use_double_quant=True,
    )

    
    # LOAD MODEL
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bits_config,
        token=hf_access_token
    ).to(device)
    print(f"Successfully loaded {model.name_or_path} onto device: {device}")


    

if __name__ == "__main__":
    main()