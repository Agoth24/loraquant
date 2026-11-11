from jaxtyping import Float
from torch import Tensor
from lq.method_utils import svd_split, compute_h, ste_optimize, quantize


def loraquant(
    B: Float[Tensor, "m r"],
    A: Float[Tensor, "r n"],
    rho: float,
    h_bits: int,
    l_bits: int,
    opt_steps: int,
    lr: float,
    group_size: int = 128,
):
    m, n, r = B.shape[0], A.shape[1], B.shape[1]
    # compute B' and A'
    B_p, A_p, S = svd_split(B, A)


    h = compute_h(rho, S)

    # split the adapters into low and high
    B_h = B_p[:, 0:h]
    A_h = A_p[0:h, :]
    
    B_l = B_p[:, h:]
    A_l = A_p[h:, :]

    
    # run STE optimization on adapters (to lower quantization error)
    for i in range(h):
        B_h[:, i: i+1], A_h[i:i+1, :] = ste_optimize(B=B_h[:, i: i+1], A=A_h[i:i+1, :], n_bits=h_bits, method="rtn", n_steps=opt_steps, lr=lr, group_size=group_size)

    
    for i in range(r-h):
        B_l[:, i:i+1], A_l[i:i+1, :] = ste_optimize(B=B_l[:, i:i+1], A=A_l[i:i+1, :], n_bits=l_bits, method="binary", n_steps=opt_steps, lr=lr, group_size=group_size)


    # quantize the high bit adapters
    Bh_q = quantize(A=B_h, n_bits=h_bits, group_size=group_size, method="rtn", columnwise=True)
    Ah_q = quantize(A=A_h, n_bits=h_bits, group_size=group_size, method="rtn", columnwise=False)
    
    # quantize the low bit adapters
    Bl_q = quantize(A=B_l, n_bits=l_bits, group_size=group_size, method="binary", columnwise=True)
    Al_q = quantize(A=A_l, n_bits=l_bits, group_size=group_size, method="binary", columnwise=False)

    # return adapters in tuples
    return (Bh_q, Ah_q), (Bl_q, Al_q)