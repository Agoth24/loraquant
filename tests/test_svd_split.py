"""svd_split — paper §3.1, Eq. 1-2."""
import torch

from conftest import make_lora
from lq.method_utils import svd_split


def test_shapes():
    B, A = make_lora(64, 8, 32)
    B_p, A_p, S = svd_split(B, A)
    assert B_p.shape == (64, 8)
    assert A_p.shape == (8, 32)
    assert S.shape == (8,)


def test_reconstructs_adapter():
    # Eq. 2: B'A' = U S V^T = BA
    B, A = make_lora(64, 8, 32)
    B_p, A_p, _ = svd_split(B, A)
    torch.testing.assert_close(B_p @ A_p, B @ A, rtol=1e-5, atol=1e-7)


def test_singular_values_sorted_and_nonnegative():
    B, A = make_lora(64, 8, 32)
    _, _, S = svd_split(B, A)
    assert (S >= 0).all()
    assert (S[:-1] >= S[1:]).all()


def test_sqrt_s_split_evenly_between_factors():
    # Eq. 2: B' = U S^1/2, A' = S^1/2 V^T, with U, V orthonormal
    # => B'^T B' = S and A' A'^T = S (as diagonal matrices)
    B, A = make_lora(64, 8, 32)
    B_p, A_p, S = svd_split(B, A)
    torch.testing.assert_close(B_p.T @ B_p, torch.diag(S), rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(A_p @ A_p.T, torch.diag(S), rtol=1e-4, atol=1e-7)


def test_rank_deficient_adapter():
    # true rank 3 < r = 8: trailing singular values should be ~0
    B, A = make_lora(64, 8, 32)
    B[:, 3:] = 0
    B_p, A_p, S = svd_split(B, A)
    assert (S[3:] < 1e-6 * S[0]).all()
    torch.testing.assert_close(B_p @ A_p, B @ A, rtol=1e-5, atol=1e-7)


def test_half_precision_input():
    # torch.linalg.svd does not support bf16; function should upcast
    B, A = make_lora(64, 8, 32)
    B_p, A_p, S = svd_split(B.bfloat16(), A.bfloat16())
    assert B_p.dtype == A_p.dtype == S.dtype == torch.float32


def test_realistic_layer_shape():
    # rank-16 adapter on a 4096-dim layer (Llama-2-7B, paper §4.1)
    B, A = make_lora(4096, 16, 4096)
    B_p, A_p, _ = svd_split(B, A)
    torch.testing.assert_close(B_p @ A_p, B @ A, rtol=1e-4, atol=1e-7)
