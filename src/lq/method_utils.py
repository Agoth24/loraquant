# mathematical functions
import torch; from torch import Tensor
import numpy
from jaxtyping import Float


def SVD_split(B: Float[Tensor, "m r"], A: Float[Tensor, "r n"],):
    delta_W = B @ A
    U: Tensor; S: Tensor; V_t: Tensor
    U, S, V_t = torch.linalg.svd(delta_W, full_matrices=False)

    B_p: Tensor = U @ S.sqrt()
    A_p: Tensor = S.sqrt() @ V_t

    assert torch.equal(B_p @ A_p, delta_W)

    return B_p, A_p


def STE_optimize(B: Float[Tensor, "m h"], A: Float[Tensor, "h n"], n_bits: int, n_steps: int, lr: float):
    
    B_opt, A_opt = B, A

    # run STE training loop over optimized matrices to minimize quantization error
    for t in range(n_steps):
        pass
        # QUANTIZE
        #DEQUANTIZE

        # COMPUTE LOSS
        # BACKPROPAGATE


    return B_opt, A_opt

def quantize():
    pass


def binary_quant():
    pass


def RTN_quant():
    pass

