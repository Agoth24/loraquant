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
    return h
    
    

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

def quantize(A: Float [Tensor, "m n"], n_bits: int, group_size: int, method: str):

    if method == "rtn":
        return rtn_quant(A, n_bits, group_size)
    elif method == "binary":
        return binary_quant(A, group_size)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    


def binary_quant(A: Float [Tensor, "m n"], group_size: int):
    m, n = A.shape

    assert n % group_size == 0, f"n = {n} must be divisible by group_size = {group_size}"

    # SPLIT INTO GROUPS
    # A: (m, n/g, g)
    A_grouped = A.reshape(m, n // group_size, group_size)

    Q = torch.sign(A_grouped)
    scale = A_grouped.abs().mean(dim=-1, keepdim=True)

    return Q, scale

def binary_dequant(Q, scale):
    return (scale * Q).reshape(Q.shape[0], -1)

# asymmetric by default
def rtn_quant(A: Float [Tensor, "m n"], n_bits: int, group_size: int):
    q_min, q_max = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1

    m, n = A.shape

    assert n % group_size == 0, f"n = {n} must be divisible by group_size = {group_size}"

    # SPLIT INTO GROUPS
    # A: (m, n/g, g)
    A_grouped = A.reshape(m, n // group_size, group_size)

    A_max = A_grouped.amax(dim=-1, keepdim=True)
    A_min = A_grouped.amin(dim=-1, keepdim=True)
    
    matrix_range = A_max - A_min

    quantized_range = q_max - q_min

    scale = (matrix_range / quantized_range).clamp(min=1e-8)

    zero_point_offset = torch.round(q_min - A_min / scale)

    Q = torch.clamp(torch.round(A_grouped / scale) + zero_point_offset, q_min, q_max)

    return Q, scale, zero_point_offset


def rtn_dequant(Q, scale, zero_point):
    return (scale * (Q - zero_point)).reshape(Q.shape[0], -1)