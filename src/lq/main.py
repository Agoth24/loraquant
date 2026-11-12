from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from lq.cli import get_args, format_results_path
from lq.adapter import quantize_adapter
from lq.utils import setup_logging, elapsed
import logging
import torch
import json
import time
import os

logger = logging.getLogger(__name__)


def main():
    # get command-line arguments
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup_logging()

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

    avg_bits = None

    # quantize the adapters
    if args.method != "fp":
        avg_bits = quantize_adapter(adapted_model, args)

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
        key: val for key, val in eval_results["results"][args.dataset].items()
        if isinstance(val, float) and "stderr" not in key
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
