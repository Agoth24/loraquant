"""svd_split — paper §3.1, Eq. 1-2."""
import pytest
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


# ---------------------------------------------------------------- method="qr"
# QR of B and A^T, then SVD of the r x r core: same contract as method="full"

def reference_svd(B, A):
    """Exact top-r singular values of BA (CPU float64)."""
    r = B.shape[1]
    return torch.linalg.svdvals(B.double() @ A.double())[:r].float()


def test_unknown_method_raises():
    B, A = make_lora(64, 8, 32)
    with pytest.raises(ValueError):
        svd_split(B, A, method="nope")


@pytest.mark.parametrize("method", ["full", "qr"])
def test_method_contract(method):
    # shapes, dtype, sorted non-negative S, and B'^T B' = A' A'^T = diag(S)
    B, A = make_lora(64, 8, 32)
    B_p, A_p, S = svd_split(B, A, method=method)
    assert B_p.shape == (64, 8) and A_p.shape == (8, 32) and S.shape == (8,)
    assert B_p.dtype == A_p.dtype == S.dtype == torch.float32
    assert (S >= 0).all() and (S[:-1] >= S[1:]).all()
    torch.testing.assert_close(B_p.T @ B_p, torch.diag(S), rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(A_p @ A_p.T, torch.diag(S), rtol=1e-4, atol=1e-7)


def test_qr_reconstructs_adapter():
    B, A = make_lora(64, 8, 32)
    B_p, A_p, _ = svd_split(B, A, method="qr")
    torch.testing.assert_close(B_p @ A_p, B @ A, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("shape", [(64, 8, 32), (4096, 16, 4096), (11008, 16, 4096), (4096, 16, 11008)])
def test_qr_singular_values_match_reference(shape):
    # includes the Llama-2-7B MLP shapes where cuda's default full svd lost accuracy
    B, A = make_lora(*shape)
    _, _, S = svd_split(B, A, method="qr")
    torch.testing.assert_close(S, reference_svd(B, A), rtol=1e-5, atol=1e-8)


def test_qr_matches_full_up_to_sign():
    # singular vectors are unique only up to sign, so compare rank-1 pieces, not factors
    B, A = make_lora(64, 8, 32)
    Bf, Af, Sf = svd_split(B, A, method="full")
    Bq, Aq, Sq = svd_split(B, A, method="qr")
    torch.testing.assert_close(Sq, Sf, rtol=1e-5, atol=1e-8)
    for i in range(B.shape[1]):
        torch.testing.assert_close(Bq[:, i:i + 1] @ Aq[i:i + 1], Bf[:, i:i + 1] @ Af[i:i + 1],
                                   rtol=1e-4, atol=1e-6, msg=f"rank-1 piece {i}")


def test_qr_rank_deficient_adapter():
    # true rank 3 < r = 8: R_B is singular, trailing singular values should be ~0
    B, A = make_lora(64, 8, 32)
    B[:, 3:] = 0
    B_p, A_p, S = svd_split(B, A, method="qr")
    assert (S[3:] < 1e-6 * S[0]).all()
    torch.testing.assert_close(B_p @ A_p, B @ A, rtol=1e-5, atol=1e-7)


def test_qr_half_precision_input():
    B, A = make_lora(64, 8, 32)
    B_p, A_p, S = svd_split(B.bfloat16(), A.bfloat16(), method="qr")
    assert B_p.dtype == A_p.dtype == S.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(4096, 16, 4096), (11008, 16, 4096)])
def test_qr_accurate_on_cuda(shape):
    # regression: full svd on cuda gave rel_err ~1.6e-3 on (11008, 16, 4096)
    B, A = make_lora(*shape)
    B, A = B.cuda(), A.cuda()
    B_p, A_p, S = svd_split(B, A, method="qr")
    assert B_p.is_cuda and A_p.is_cuda and S.is_cuda
    rel_err = (B_p @ A_p - B @ A).norm() / (B @ A).norm()
    assert rel_err < 1e-5, rel_err
    torch.testing.assert_close(S.cpu(), reference_svd(B.cpu(), A.cpu()), rtol=1e-4, atol=1e-8)
