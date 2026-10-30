"""Quantizers — paper §3.2 (RTN Eq. 6-8, binary Eq. 9-10) and STE §3.3."""
import pytest
import torch

from lq.method_utils import (
    binary_dequant,
    binary_quant,
    quantize,
    rtn_dequant,
    rtn_quant,
    ste_round,
)


# ---------------------------------------------------------------- ste_round

class TestSteRound:
    def test_forward_is_round(self):
        x = torch.randn(8, 8) * 5
        torch.testing.assert_close(ste_round(x), torch.round(x))

    def test_backward_is_identity(self):
        x = (torch.randn(8, 8) * 5).requires_grad_()
        ste_round(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------- RTN

class TestRtn:
    @pytest.mark.parametrize("n_bits", [2, 3, 4, 8])
    def test_codes_are_integers_in_range(self, n_bits):
        A = torch.randn(16, 64)
        Q, _, _ = rtn_quant(A, n_bits, group_size=16)
        q_min, q_max = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
        assert (Q == Q.round()).all()
        assert Q.min() >= q_min and Q.max() <= q_max

    def test_dequant_shape_matches_input(self):
        A = torch.randn(16, 64)
        assert rtn_dequant(*rtn_quant(A, 2, group_size=16)).shape == A.shape

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_error_bounded_by_half_step(self, n_bits):
        # Eq. 6-8: within the group range, |A - D(Q(A))| <= S/2
        A = torch.randn(16, 64)
        g = 16
        Q, scale, zero = rtn_quant(A, n_bits, group_size=g)
        err = (A - rtn_dequant(Q, scale, zero)).abs().reshape(16, 64 // g, g)
        assert (err <= scale / 2 * (1 + 1e-4)).all()

    def test_more_bits_less_error(self):
        A = torch.randn(16, 64)
        errs = [
            (A - rtn_dequant(*rtn_quant(A, b, group_size=16))).norm()
            for b in [2, 3, 4, 8]
        ]
        assert errs == sorted(errs, reverse=True)

    def test_groups_are_independent(self):
        # changing group 2 must not change the reconstruction of group 1
        A = torch.randn(4, 32)
        A2 = A.clone()
        A2[:, 16:] *= 100
        D1 = rtn_dequant(*rtn_quant(A, 2, group_size=16))
        D2 = rtn_dequant(*rtn_quant(A2, 2, group_size=16))
        torch.testing.assert_close(D1[:, :16], D2[:, :16])

    def test_constant_group_no_nan(self):
        A = torch.ones(4, 16)
        D = rtn_dequant(*rtn_quant(A, 2, group_size=16))
        assert torch.isfinite(D).all()
        torch.testing.assert_close(D, A, rtol=1e-3, atol=1e-3)

    def test_group_size_must_divide_n(self):
        with pytest.raises(AssertionError):
            rtn_quant(torch.randn(4, 30), 2, group_size=16)

    def test_gradient_flows_via_ste(self):
        A = torch.randn(4, 32).requires_grad_()
        rtn_dequant(*rtn_quant(A, 2, group_size=16)).sum().backward()
        assert A.grad is not None and A.grad.abs().sum() > 0


# ---------------------------------------------------------------- binary

class TestBinary:
    def test_codes_are_plus_minus_one(self):
        A = torch.randn(16, 64)
        Q, _, _ = binary_quant(A, group_size=16)
        assert set(Q.unique().tolist()) <= {-1.0, 1.0}

    def test_sign_of_zero_is_plus_one(self):
        # Eq. 9: sign(x) = 1 if x >= 0
        A = torch.randn(4, 16)
        A[0, 0] = 0.0
        Q, _, _ = binary_quant(A, group_size=16)
        assert Q.reshape(4, 16)[0, 0] == 1.0

    def test_scale_is_mean_abs_per_group(self):
        # Eq. 10
        A = torch.randn(4, 32)
        _, scale, _ = binary_quant(A, group_size=16)
        expected = A.reshape(4, 2, 16).abs().mean(-1, keepdim=True)
        torch.testing.assert_close(scale, expected)

    def test_scale_minimizes_frobenius_error(self):
        # Eq. 10 is optimal for fixed signs (Rastegari et al., 2016)
        A = torch.randn(4, 16)
        Q, scale, _ = binary_quant(A, group_size=16)
        best = (A - binary_dequant(Q, scale)).norm()
        for f in [0.9, 1.1]:
            assert (A - binary_dequant(Q, scale * f)).norm() > best

    def test_dequant_reconstruction(self):
        A = torch.randn(4, 32)
        D = binary_dequant(*binary_quant(A, group_size=16))
        assert D.shape == A.shape
        expected = torch.sign(A) * A.reshape(4, 2, 16).abs().mean(-1, keepdim=True).repeat_interleave(16, dim=1).reshape(4, 32)
        torch.testing.assert_close(D, expected)

    def test_groups_are_independent(self):
        A = torch.randn(4, 32)
        A2 = A.clone()
        A2[:, 16:] *= 100
        D1 = binary_dequant(*binary_quant(A, group_size=16))
        D2 = binary_dequant(*binary_quant(A2, group_size=16))
        torch.testing.assert_close(D1[:, :16], D2[:, :16])

    def test_group_size_must_divide_n(self):
        with pytest.raises(AssertionError):
            binary_quant(torch.randn(4, 30), group_size=16)


# ---------------------------------------------------------------- quantize dispatch

class TestQuantizeDispatch:
    def test_rtn(self):
        A = torch.randn(4, 32)
        for got, exp in zip(quantize(A, 2, 16, "rtn"), rtn_quant(A, 2, 16)):
            torch.testing.assert_close(got, exp)

    def test_binary(self):
        A = torch.randn(4, 32)
        for got, exp in zip(quantize(A, 1, 16, "binary"), binary_quant(A, 16)):
            torch.testing.assert_close(got, exp)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            quantize(torch.randn(4, 32), 2, 16, "nf4")
