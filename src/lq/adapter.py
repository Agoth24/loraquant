from peft import get_peft_model_state_dict
from lq.method_utils import quantize, dequantize
from lq.loraquant import loraquant
from lq.utils import elapsed
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
import argparse
import logging
import torch
import time
import sys

logger = logging.getLogger(__name__)


# quantize every LoRA layer of `adapted_model` in place; returns average bits per adapter weight
def quantize_adapter(adapted_model, args: argparse.Namespace) -> float:
    # check for STE optimization
    opt_steps = args.opt_steps if args.opt else 0

    lora_state_dict = get_peft_model_state_dict(adapted_model)

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

    return avg_bits
