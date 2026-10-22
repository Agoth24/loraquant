from jaxtyping import Float
from torch import Tensor


def loraquant(
    B: Float[Tensor, "m r"],
    A: Float[Tensor, "r n"],
    rho: float,
    h_bits: int,
    l_bits: int,
    opt_steps: int,
    lr: float,
):
    pass
