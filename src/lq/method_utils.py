# mathematical functions
import torch; from torch import Tensor
from jaxtyping import Float


def svd_split(B: Float[Tensor, "m r"], A: Float[Tensor, "r n"]):

    # get rank
    rank = B.shape[1]

    # decompose w/ SVD
    delta_W = B.float() @ A.float()
    U, S, V_t = torch.linalg.svd(delta_W, full_matrices=False)
    
    # truncate the matrices
    U, S, V_t = U[:, :rank], S[:rank], V_t[:rank, :]

    B_prime, A_prime = U * S.sqrt(), S.sqrt()[:, None] * V_t

    rel_err = (B_prime @ A_prime - delta_W).norm() / delta_W.norm()
    
    # assert the matrices are close in working precision
    assert rel_err < 1e-6, rel_err

    return B_prime, A_prime, S

def compute_h(rho: float, S: Float[Tensor, "r"]) -> int:
    # energy is the sum of squared singular values
    energy = S ** 2

    candidate_var = energy.cumsum(0)
    total_var = energy.sum()

    explained = candidate_var / total_var

    candidate_indices = (explained >= rho).nonzero(as_tuple=True)[0]
    if len(candidate_indices) == 0:
        raise ValueError(f"rho = {rho} is not attained by any number of singular values")

    h = candidate_indices[0].item() + 1
    

    
    

def STE_optimize(B: Float[Tensor, "m h"], A: Float[Tensor, "h n"], n_bits: int, n_steps: int, lr: float):
    
    B_opt, A_opt = B, A

    # run STE training loop over optimized matrices to minimize quantization error
    for t in range(n_steps):
        pass
        # QUANTIZE
        # DEQUANTIZE

        # COMPUTE LOSS
        # BACKPROPAGATE


    return B_opt, A_opt

def quantize():
    pass


def binary_quant():
    pass


def RTN_quant():
    pass

