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
    assert rel_err < 1e-6, rel_err

    return B_prime, A_prime, S


def compute_h(rho: float, S: Float[Tensor, "r"]) -> int:
    # energy is the sum of squared singular values
    energy = S**2

    cumulative_variance = energy.cumsum(0)
    overall_variance = energy.sum()

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


def ste_optimize(
    B: Float[Tensor, "m h"],
    A: Float[Tensor, "h n"],
    n_bits: int,
    n_steps: int,
    lr: float,
):
    B_opt, A_opt = (
        B.detach().clone().requires_grad_(),
        A.detach().clone().requires_grad_(),
    )

    training_target = (B @ A).detach()

    # init optimizer for B_opt, A_opt
    optimizer = torch.optim.AdamW([B_opt, A_opt], lr=lr, weight_decay=0)

    for _ in range(n_steps):
        # quantize
        Q_B, scale_B, zero_B = rtn_quant(B_opt, n_bits, group_size=1)
        Q_A, scale_A, zero_A = rtn_quant(A_opt, n_bits, group_size=1)

        # dequantize (recover the matrices)
        B_rec = ste_round(rtn_dequant(Q_B, scale_B, zero_B))
        A_rec = ste_round(rtn_dequant(Q_A, scale_A, zero_A))

        # compute the frobenius norm between recovered & original adapter
        loss = torch.norm(B_rec @ A_rec - training_target, p="fro")

        # BACKPROPAGATE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # detach from the computation graph
    return B_opt.detach(), A_opt.detach()


def quantize(A: Float[Tensor, "m n"], n_bits: int, group_size: int, method: str):
    if method == "rtn":
        return rtn_quant(A, n_bits, group_size)
    elif method == "binary":
        return binary_quant(A, group_size)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def binary_quant(A: Float[Tensor, "m n"], group_size: int):
    m, n = A.shape

    assert (
        n % group_size == 0
    ), f"n = {n} must be divisible by group_size = {group_size}"

    # SPLIT INTO GROUPS
    # A: (m, n/g, g)
    A_grouped = A.reshape(m, n // group_size, group_size)

    Q = torch.sign(A_grouped)
    scale = A_grouped.abs().mean(dim=-1, keepdim=True)

    return Q, scale


def binary_dequant(Q, scale):
    return (scale * Q).reshape(Q.shape[0], -1)


# asymmetric by default
def rtn_quant(A: Float[Tensor, "m n"], n_bits, group_size: int):
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


def rtn_dequant(Q, scale, zero_point):
    return (scale * (Q - zero_point)).reshape(Q.shape[0], -1)
