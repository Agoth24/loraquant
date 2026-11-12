from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model_state_dict
from lm_eval.models.huggingface import HFLM
from lq.method_utils import quantize, dequantize
from lq.loraquant import loraquant
from transformers.utils import logging as hf_logging
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
from lm_eval import evaluator
from pathlib import Path
from typing import Any
import argparse
import logging
import torch
import json
import time
import sys
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
        "hi": args.high_bits,
        "lo": args.low_bits if uses_low_bits else None,
        "grp": args.group_size if quantized else None,
        "rho": args.ratio if args.method == "lq" else None,  # ratio is only used by lq
        "opt": args.opt_steps if opt else None,
        "lr": args.lr if opt else None,
        "lim": args.limit,
    }

    filename_parts = [args.method] + [
        f"{opt}{val}" for opt, val in options.items() if val is not None
    ]

    filename = "_".join(filename_parts) + ".json"

    return Path(args.res_path) / model_name / args.dataset / args.method / filename


logger = logging.getLogger(__name__)


# elapsed time since `start` as e.g. "4m56s"
def elapsed(start: float) -> str:
    m, s = divmod(int(time.perf_counter() - start), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main():
    # get command-line arguments
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # silence library chatter: only warnings and errors from transformers / lm_eval
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("lm_eval").setLevel(logging.WARNING)

    # check for STE optimization
    opt_steps = args.opt_steps if args.opt else 0

    out_path = format_results_path(args)

    # one-line run summary: the results filename already encodes the method's settings
    logger.info("%s | %s | %s | %s", out_path.stem, args.model.split("/")[-1], args.dataset, device)

    t_load = time.perf_counter()

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
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        quantization_config=bits_config,
        token=hf_access_token,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model, token=hf_access_token
    )

    # add the pre-trained adapter to the model
    adapted_model = PeftModel.from_pretrained(
        model=base_model, model_id=args.adapter_path, local_files_only=True
    )
    logger.info("loaded model + adapter (%s)", elapsed(t_load))

    adapted_model.eval()

    lora_state_dict = get_peft_model_state_dict(adapted_model)

    avg_bits = None

    # quantize the adapters
    if args.method != "fp":
        # one entry per layer: visit each layer once via its A key (its B is found from the name)
        A_keys = [name for name in lora_state_dict if name.endswith(".lora_A.weight")]

        t_quant = time.perf_counter()
        total_params, total_bits = 0, 0

        with torch.no_grad(), logging_redirect_tqdm():
            for name in tqdm(A_keys, desc="quantizing", unit="layer", file=sys.stdout):
                A_name = name.replace(".lora_A.weight", ".lora_A.default")
                B_name = name.replace(".lora_A.weight", ".lora_B.default")

                lora_B = adapted_model.get_submodule(B_name).weight
                lora_A = adapted_model.get_submodule(A_name).weight

                # quantize the adapter weights
                if args.method in ["rtn", "bin"]:
                    q_method = "rtn" if args.method == "rtn" else "binary"
                    n_bits = args.high_bits if args.method == "rtn" else args.low_bits

                    qB = quantize(A=lora_B, n_bits=n_bits, group_size=args.group_size, method=q_method, columnwise=True)
                    qA = quantize(A=lora_A, n_bits=n_bits, group_size=args.group_size, method=q_method, columnwise=False)

                    B = dequantize(*qB, method=q_method, columnwise=True)
                    A = dequantize(*qA, method=q_method, columnwise=False)

                elif args.method == "lq":
                    (Bh_q, Ah_q), (Bl_q, Al_q) = loraquant(B=lora_B, A=lora_A, rho=args.ratio, h_bits=args.high_bits, l_bits=args.low_bits, opt_steps=opt_steps, lr=args.lr, group_size=args.group_size)

                    B = torch.cat([dequantize(*Bh_q, method="rtn", columnwise=True), dequantize(*Bl_q, method="binary", columnwise=True)], dim=1)
                    A = torch.cat([dequantize(*Ah_q, method="rtn", columnwise=False), dequantize(*Al_q, method="binary", columnwise=False)], dim=0)

                # logic for computing bits
                m, r, n = lora_B.shape[0], lora_B.shape[1], lora_A.shape[1]

                if args.method == "lq":
                    h = Bh_q[0].shape[0]  # high B codes are (h, m/g, g)
                    total_bits += (h * args.high_bits + (r - h) * 1) * (m + n)
                else:
                    bits_used = args.high_bits if args.method == "rtn" else 1
                    total_bits += r * (m + n) * bits_used

                total_params += r * (m + n)

                # cast back to the original dtype; .T from columnwise dequant isn't contiguous
                lora_B.data = B.to(lora_B.dtype).contiguous()
                lora_A.data = A.to(lora_A.dtype).contiguous()

        avg_bits = total_bits / total_params
        logger.info("quantized %d layers (%s) | avg bits %.3f", len(A_keys), elapsed(t_quant), avg_bits)

    # evaluation section
    logger.info("eval %s%s", args.dataset, f" (limit {args.limit})" if args.limit else "")
    t_eval = time.perf_counter()

    hf_lm = HFLM(pretrained=adapted_model, tokenizer=tokenizer, device=device, batch_size="auto")

    eval_results = evaluator.simple_evaluate(
        model=hf_lm,
        tasks=[args.dataset],
        num_fewshot=0,  # no fewshots given
        limit=args.limit,
        batch_size="auto",
        device=device,
    )
    # only log numeric metrics
    metrics = {
        k: v for k, v in eval_results["results"][args.dataset].items()
        if isinstance(v, float) and "stderr" not in k
    }
    logger.info(
        "%s | %s (%s)",
        args.dataset, " | ".join(f"{k} {v:.3f}" for k, v in metrics.items()), elapsed(t_eval),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # log eval results to json
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "results": eval_results["results"],
                "avg_bits": avg_bits,
                "generations": eval_results["samples"][args.dataset],
            },
            f,
            indent=2,
        )
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
