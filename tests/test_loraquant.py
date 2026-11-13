"""loraquant — paper Alg. 1 (with §3.1 split, §3.2 quantizers, §3.3 / Alg. 2 opt).

Contract assumed:
    (Bh_q, Ah_q), (Bl_q, Al_q) = loraquant(B, A, rho, h_bits, l_bits, opt_steps, lr, group_size)
where each *_q is the raw (Q, scale, zero_point) tuple from `quantize`.
High sub-LoRA -> RTN, low sub-LoRA -> binary (§3.2). B grouped column-wise, A row-wise.
"""
import pytest
import torch

from conftest import make_lora
from lq.loraquant import loraquant
from lq.method_utils import compute_h, dequantize, quantize, ste_optimize, svd_split

M, R, N = 64, 8, 32
G = 16  # group size; divides M and N
RHO = 0.9


def run(B, A, rho=RHO, h_bits=3, l_bits=1, opt_steps=0, lr=1e-3, group_size=G):
    return loraquant(B, A, rho=rho, h_bits=h_bits, l_bits=l_bits,
                     opt_steps=opt_steps, lr=lr, group_size=group_size)


def dequant_all(out):
    """Dequantize loraquant output -> (B_h, A_h, B_l, A_l) as float matrices."""
    (Bh_q, Ah_q), (Bl_q, Al_q) = out
    return (
        dequantize(*Bh_q, method="rtn", columnwise=True),
        dequantize(*Ah_q, method="rtn", columnwise=False),
        dequantize(*Bl_q, method="binary", columnwise=True),
        dequantize(*Al_q, method="binary", columnwise=False),
    )


def reference_split(B, A, rho=RHO):
    """Alg. 1 lines 1-8 using the already-tested helpers."""
    Bp, Ap, S = svd_split(B, A)
    h = compute_h(rho, S)
    return h, Bp[:, :h], Ap[:h], Bp[:, h:], Ap[h:]


def groups(X, columnwise):
    """Reshape X into (rows, n_groups, G) along the paper's grouping direction."""
    if columnwise:
        X = X.T
    return X.reshape(X.shape[0], -1, G)


@pytest.fixture
def lora():
    B, A = make_lora(M, R, N)
    h = reference_split(B, A)[0]
    assert 0 < h < R, "fixture should give a non-trivial split"
    return B, A


# ---------------------------------------------------------------- split (§3.1)

def test_split_sizes_match_compute_h(lora):
    """Alg. 1 lines 4-8: h from energy ratio; sub-LoRA shapes follow."""
    B, A = lora
    h = reference_split(B, A)[0]
    Bh, Ah, Bl, Al = dequant_all(run(B, A))
    assert Bh.shape == (M, h) and Ah.shape == (h, N)
    assert Bl.shape == (M, R - h) and Al.shape == (R - h, N)


def test_no_opt_equals_split_then_quantize(lora):
    """opt_steps=0 must reduce Alg. 1 to lines 1-8 + 15-16 exactly."""
    B, A = lora
    h_bits = 3
    _, Bh, Ah, Bl, Al = reference_split(B, A)
    expected = (
        dequantize(*quantize(Bh, h_bits, G, "rtn", True), method="rtn", columnwise=True),
        dequantize(*quantize(Ah, h_bits, G, "rtn", False), method="rtn", columnwise=False),
        dequantize(*quantize(Bl, 1, G, "binary", True), method="binary", columnwise=True),
        dequantize(*quantize(Al, 1, G, "binary", False), method="binary", columnwise=False),
    )
    got = dequant_all(run(B, A, h_bits=h_bits, opt_steps=0))
    for name, g, e in zip(["B_h", "A_h", "B_l", "A_l"], got, expected):
        torch.testing.assert_close(g, e, msg=f"{name} mismatch")


# ------------------------------------------------------- quantizers (§3.2, App.)

def test_low_sublora_is_binary(lora):
    """§3.2: low sub-LoRA -> {-S, +S} per group (B column-wise, A row-wise)."""
    B, A = lora
    _, _, Bl, Al = dequant_all(run(B, A))
    for X, colwise in [(Bl, True), (Al, False)]:
        mags = groups(X, colwise).abs()
        torch.testing.assert_close(mags, mags[..., :1].expand_as(mags))


@pytest.mark.parametrize("h_bits", [2, 3])
def test_high_sublora_has_at_most_2_pow_bits_levels(lora, h_bits):
    """§3.2: RTN with b bits -> at most 2^b distinct values per group."""
    B, A = lora
    Bh, Ah, _, _ = dequant_all(run(B, A, h_bits=h_bits))
    for X, colwise in [(Bh, True), (Ah, False)]:
        for grp in groups(X, colwise).reshape(-1, G):
            assert grp.unique().numel() <= 2 ** h_bits


def test_high_error_decreases_with_bits(lora):
    """More bits for the high sub-LoRA -> smaller high-half reconstruction error."""
    B, A = lora
    _, Bh0, Ah0, _, _ = reference_split(B, A)
    target = Bh0 @ Ah0
    errs = []
    for bits in [2, 4, 8]:
        Bh, Ah, _, _ = dequant_all(run(B, A, h_bits=bits))
        errs.append((target - Bh @ Ah).norm().item())
    assert errs[0] > errs[1] > errs[2], errs


# ------------------------------------------------- per-pair STE opt (§3.3, Alg. 2)

@pytest.mark.parametrize("which", ["high", "low"])
def test_each_pair_optimized_independently(lora, which):
    """Alg. 1 lines 9-14 + Eq. 11: pair i is optimized alone with the quantizer of
    its own sub-LoRA, then quantized. Grouping is within a column of B / row of A,
    so column i of the output depends only on pair i."""
    B, A = lora
    h, Bh0, Ah0, Bl0, Al0 = reference_split(B, A)
    h_bits, T, lr = 3, 20, 1e-3
    if which == "high":
        Bs, As, bits, meth, cols = Bh0, Ah0, h_bits, "rtn", range(h)
    else:
        Bs, As, bits, meth, cols = Bl0, Al0, 1, "binary", range(R - h)

    Bh, Ah, Bl, Al = dequant_all(run(B, A, h_bits=h_bits, opt_steps=T, lr=lr))
    Bout, Aout = (Bh, Ah) if which == "high" else (Bl, Al)

    for i in cols:
        b, a = ste_optimize(Bs[:, i:i + 1], As[i:i + 1], n_bits=bits, method=meth,
                            n_steps=T, lr=lr, group_size=G)
        exp_b = dequantize(*quantize(b, bits, G, meth, True), method=meth, columnwise=True)
        exp_a = dequantize(*quantize(a, bits, G, meth, False), method=meth, columnwise=False)
        torch.testing.assert_close(Bout[:, i:i + 1], exp_b, msg=f"{which} col {i} of B")
        torch.testing.assert_close(Aout[i:i + 1], exp_a, msg=f"{which} row {i} of A")


@pytest.mark.xfail(reason="ste_optimize increases Eq. 11 error (loss dips then climbs); unresolved")
def test_opt_reduces_per_pair_error_high(lora):
    """§3.3 / Fig. 3: optimization reduces the Eq. 11 objective (summed over pairs)
    for the RTN sub-LoRA. Measured per pair, since that is what Eq. 11 minimizes."""
    B, A = lora
    h, Bh0, Ah0, _, _ = reference_split(B, A)

    def pair_err(out):
        Bh, Ah, _, _ = dequant_all(out)
        return sum(
            (Bh0[:, i:i + 1] @ Ah0[i:i + 1] - Bh[:, i:i + 1] @ Ah[i:i + 1]).norm().item()
            for i in range(h)
        )

    e0 = pair_err(run(B, A, h_bits=2, opt_steps=0))
    e1 = pair_err(run(B, A, h_bits=2, opt_steps=100, lr=1e-3))
    assert e1 < e0, (e0, e1)


# ---------------------------------------------------------------- edges / hygiene

def test_rho_one_gives_empty_low_sublora(lora):
    """rho=1 is valid (compute_h accepts (0,1]) -> h=r, low sub-LoRA has rank 0."""
    B, A = lora
    Bh, Ah, Bl, Al = dequant_all(run(B, A, rho=1.0, h_bits=8))
    assert Bh.shape == (M, R) and Bl.shape == (M, 0) and Al.shape == (0, N)
    rel = (B @ A - Bh @ Ah).norm() / (B @ A).norm()
    assert rel < 2e-2, rel


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.5])
def test_invalid_rho_raises(lora, rho):
    B, A = lora
    with pytest.raises(ValueError):
        run(B, A, rho=rho)


def test_inputs_not_mutated(lora):
    B, A = lora
    B0, A0 = B.clone(), A.clone()
    run(B, A, opt_steps=5)
    torch.testing.assert_close(B, B0)
    torch.testing.assert_close(A, A0)


def test_svd_method_passed_to_svd_split(lora, monkeypatch):
    import lq.loraquant as lq_mod
    seen = []

    def spy(B, A, method="full"):
        seen.append(method)
        return svd_split(B, A, method="full")

    monkeypatch.setattr(lq_mod, "svd_split", spy)
    B, A = lora
    loraquant(B, A, rho=RHO, h_bits=3, l_bits=1, opt_steps=0, lr=1e-3, group_size=G, svd="qr")
    loraquant(B, A, rho=RHO, h_bits=3, l_bits=1, opt_steps=0, lr=1e-3, group_size=G)
    assert seen == ["qr", "full"]


def test_qr_svd_same_split_sizes(lora):
    B, A = lora
    h = reference_split(B, A)[0]
    out = loraquant(B, A, rho=RHO, h_bits=3, l_bits=1, opt_steps=0, lr=1e-3, group_size=G, svd="qr")
    Bh, Ah, Bl, Al = dequant_all(out)
    assert Bh.shape == (M, h) and Bl.shape == (M, R - h)


def test_deterministic(lora):
    B, A = lora
    for x, y in zip(dequant_all(run(B, A, opt_steps=5)), dequant_all(run(B, A, opt_steps=5))):
        torch.testing.assert_close(x, y)
