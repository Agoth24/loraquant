# mathematical functions
import torch
from torch import Tensor
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
    assert rel_err < 1e-4, rel_err

    return B_prime, A_prime, S


def compute_h(rho: float, S: Float[Tensor, "r"]) -> int:

    if not 0 < rho <= 1:
        raise ValueError(f"variance ratio = {rho} must be in (0, 1]")

    # energy is the sum of squared singular values
    energy = S**2

    cumulative_variance = energy.cumsum(0)
    # last element of ratio is 1
    overall_variance = cumulative_variance[-1]

    explained_variance_ratio = cumulative_variance / overall_variance

    candidate_indices = (explained_variance_ratio >= rho).nonzero(as_tuple=True)[0]

    if len(candidate_indices) == 0:
        raise ValueError(
            f"rho = {rho} is not attained by any number of singular values"
        )

    h = candidate_indices[0].item() + 1

    return h


def ste_round(x: Float[Tensor, "m n"]):
    # pytorch backward pass only sees x
    return x + (torch.round(x) - x).detach()


def ste_sign(x: Float[Tensor, "m n"]):
    return x + (torch.where(x >= 0, 1, -1) - x).detach()


def ste_optimize(
    B: Float[Tensor, "m h"],
    A: Float[Tensor, "h n"],
    n_bits: int,
    method: str,
    n_steps: int = 100,
    lr: float = 1e-2,
    group_size: int = 128,
):

    # clone the adapter matrices
    B_opt, A_opt = (
        B.detach().clone().requires_grad_(),
        A.detach().clone().requires_grad_(),
    )

    training_target = (B @ A).detach()

    # init optimizer for B_opt, A_opt
    optimizer = torch.optim.Adam([B_opt, A_opt], lr=lr)

    # run STE
    for _ in range(n_steps):
        # quantize
        qB = quantize(B_opt, n_bits, group_size, method, columnwise=True)
        qA = quantize(A_opt, n_bits, group_size, method, columnwise=False)

        # dequantize (recover the matrices)
        B_rec = dequantize(*qB, method=method, columnwise=True)
        A_rec = dequantize(*qA, method=method, columnwise=False)

        # compute the frobenius norm between recovered & original adapter
        loss = torch.norm(B_rec @ A_rec - training_target, p="fro")

        # BACKPROPAGATE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # detach from the computation graph
    return B_opt.detach(), A_opt.detach()


def quantize(
    A: Float[Tensor, "m n"],
    n_bits: int,
    group_size: int,
    method: str,
    columnwise: bool = False,
):
    if method == "rtn":
        return rtn_quant(A, n_bits, group_size, columnwise)
    elif method == "binary":
        return binary_quant(A, group_size, columnwise)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def dequantize(Q, scale, zero_point, method: str, columnwise: bool = False):
    if method == "rtn":
        return rtn_dequant(Q, scale, zero_point, columnwise)
    elif method == "binary":
        return binary_dequant(Q, scale, columnwise)
    else:
        raise ValueError(f"Unknown dequantization method {method}")


def binary_quant(A: Float[Tensor, "m n"], group_size: int, columnwise: bool = False):
    if columnwise:
        A = A.T

    m, n = A.shape

    assert (
        n % group_size == 0
    ), f"n = {n} must be divisible by group_size = {group_size}"

    # SPLIT INTO GROUPS
    # A: (m, n/g, g)
    A_grouped = A.reshape(m, n // group_size, group_size)

    scale = A_grouped.abs().mean(dim=-1, keepdim=True)
    Q = ste_sign(A_grouped)

    return Q, scale, None


def binary_dequant(Q, scale, columnwise: bool = False):
    dQ = (scale * Q).reshape(Q.shape[0], Q.shape[1] * Q.shape[2] )
    return dQ.T if columnwise else dQ


# asymmetric by default
def rtn_quant(
    A: Float[Tensor, "m n"], n_bits: int, group_size: int = 128, columnwise: bool = False
):

    if columnwise:
        A = A.T

    m, n = A.shape

    q_min, q_max = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1

    assert (
        n % group_size == 0
    ), f"n = {n} must be divisible by group_size = {group_size}"

    # SPLIT INTO GROUPS
    # A: (m, n/g, g)
    A_grouped = A.reshape(m, n // group_size, group_size)

    A_max = A_grouped.amax(dim=-1, keepdim=True)
    A_min = A_grouped.amin(dim=-1, keepdim=True)

    matrix_range = A_max - A_min
    quantized_range = q_max - q_min

    scale = (matrix_range / quantized_range).clamp(min=1e-8)
    zero_point_offset = torch.round(q_min - A_min / scale)

    Q = torch.clamp(ste_round(A_grouped / scale) + zero_point_offset, q_min, q_max)

    return Q, scale, zero_point_offset


def rtn_dequant(Q, scale, zero_point, columnwise: bool = False):
    dQ = (scale * (Q - zero_point)).reshape(Q.shape[0], Q.shape[1] * Q.shape[2])
    return dQ.T if columnwise else dQ
