from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import argparse
import os
import json
from pathlib import Path
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

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
    "--group_size": dict(type=int, default=128),
    "--high_bits": dict(type=int),
    "--low_bits": dict(type=int),
    "--ratio": dict(type=float, default=0.7),
    "--opt": dict(action="store_true"),
    "--limit": dict(type=int),
    "--res_path": dict(type=str, default="outputs"),
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

    # if the method quantizes the adapter
    quantized = args.method != "fp"

    options = {
        "hi": args.high_bits,
        "lo": args.low_bits,
        "grp": args.group_size if quantized else None,
        "rho": args.ratio if quantized else None,
        "lim": args.limit,
    }

    filename_parts = [args.method] + [
        f"{opt}{val}" for opt, val in options.items() if val is not None
    ]

    filename = "_".join(filename_parts) + ".json"

    return Path(args.res_path) / model_name / args.dataset / args.method / filename


def main():
    # set device for torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # grab CLAs and output path
    args = get_args()

    out_path = format_results_path(args)

    hf_access_token = os.environ.get("HF_TOKEN", default=None)

    compute_data_type = getattr(torch, "float16")

    # set model quantization config
    bits_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_data_type,
        bnb_4bit_use_double_quant=True,
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        quantization_config=bits_config,
        token=hf_access_token,
        device_map="auto",
    )
    print(
        f"Successfully loaded {model.name_or_path} onto device: {device.capitalize()}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model, token=hf_access_token
    )

    # add the pre-trained adapter to the model
    model = PeftModel.from_pretrained(
        model=model, model_id=args.adapter_path, local_files_only=True
    )

    # EVAL SECTION
    model.eval()

    hf_lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device)

    # run evals
    eval_results: dict = evaluator.simple_evaluate(
        model=hf_lm,
        tasks=[args.dataset],
        num_fewshot=0,  # no fewshots given
        limit=args.limit,
        batch_size="auto",
        device=device,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # log eval results to json
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "results": eval_results["results"],
                "generations": eval_results["samples"][args.dataset],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
