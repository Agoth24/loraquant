"""ste_optimize — paper §3.3, Eq. 11, Alg. 2."""
import torch

from conftest import make_lora
from lq.method_utils import rtn_dequant, rtn_quant, ste_optimize

# group size used to *measure* quantization error. ste_optimize does not take
# a group size yet; update this once it does so both sides agree.
GROUP_SIZE = 16


def quant_error(B, A, n_bits, group_size=GROUP_SIZE):
    """||BA - D(Q(B)) D(Q(A))||_F with B quantized column-wise, A row-wise
    (paper Appendix: B' column-wise, A' row-wise)."""
    B_rec = rtn_dequant(*rtn_quant(B.T, n_bits, group_size)).T
    A_rec = rtn_dequant(*rtn_quant(A, n_bits, group_size))
    return torch.norm(B @ A - B_rec @ A_rec, p="fro").item()


def test_output_shapes_and_detached():
    B, A = make_lora(64, 4, 32)
    B_opt, A_opt = ste_optimize(B, A, n_bits=2, method="rtn", n_steps=5, lr=1e-3, group_size=GROUP_SIZE)
    assert B_opt.shape == B.shape and A_opt.shape == A.shape
    assert not B_opt.requires_grad and not A_opt.requires_grad


def test_inputs_not_mutated():
    B, A = make_lora(64, 4, 32)
    B0, A0 = B.clone(), A.clone()
    ste_optimize(B, A, n_bits=2, method="rtn", n_steps=5, lr=1e-3, group_size=GROUP_SIZE)
    torch.testing.assert_close(B, B0)
    torch.testing.assert_close(A, A0)


def test_zero_steps_is_identity():
    B, A = make_lora(64, 4, 32)
    B_opt, A_opt = ste_optimize(B, A, n_bits=2, method="rtn", n_steps=0, lr=1e-3, group_size=GROUP_SIZE)
    torch.testing.assert_close(B_opt, B)
    torch.testing.assert_close(A_opt, A)


def test_reduces_quantization_error_single_pair():
    # Alg. 1 lines 9-14 call opt on one column b_i / row a_i at a time
    B, A = make_lora(64, 1, 32)
    before = quant_error(B, A, n_bits=2)
    B_opt, A_opt = ste_optimize(B, A, n_bits=2, method="rtn", n_steps=100, lr=1e-3, group_size=GROUP_SIZE)
    after = quant_error(B_opt, A_opt, n_bits=2)
    assert after < before, (before, after)


def test_reduces_quantization_error_joint():
    # footnote 3: joint optimization of a whole sub-LoRA also works
    B, A = make_lora(64, 4, 32)
    before = quant_error(B, A, n_bits=2)
    B_opt, A_opt = ste_optimize(B, A, n_bits=2, method="rtn", n_steps=100, lr=1e-3, group_size=GROUP_SIZE)
    after = quant_error(B_opt, A_opt, n_bits=2)
    assert after < before, (before, after)
