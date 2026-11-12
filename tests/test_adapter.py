"""lq.adapter.quantize_adapter — run on a tiny PEFT model (no 7B model / GPU needed).

The tiny model has two adapted Linear layers, so the key names, `.default` lookup,
in-place write-back and bit accounting are exercised exactly as on Llama.
"""
import argparse

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from lq.adapter import quantize_adapter
from lq.method_utils import compute_h, svd_split

D_IN, D_OUT, R, G = 256, 384, 8, 128  # G divides both D_IN and D_OUT
LAYERS = ["q", "v"]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(D_IN, D_OUT, bias=False)
        self.v = torch.nn.Linear(D_IN, D_OUT, bias=False)

    def forward(self, x):
        return self.q(x) + self.v(x)


@pytest.fixture
def model():
    # init_lora_weights=False -> random B (default init sets B = 0, which makes ΔW = 0)
    cfg = LoraConfig(r=R, lora_alpha=16, target_modules=LAYERS, init_lora_weights=False)
    return get_peft_model(Tiny(), cfg)


def make_args(**overrides) -> argparse.Namespace:
    base = dict(method="rtn", group_size=G, high_bits=2, low_bits=None,
                ratio=0.7, opt=False, opt_steps=10, lr=1e-3)
    return argparse.Namespace(**(base | overrides))


def lora_weights(model):
    """{layer: (B, A)} cloned from the live model."""
    return {
        name: (model.get_submodule(f"base_model.model.{name}.lora_B.default").weight.detach().clone(),
               model.get_submodule(f"base_model.model.{name}.lora_A.default").weight.detach().clone())
        for name in LAYERS
    }


def n_distinct_per_group(W, group_size, columnwise):
    """Max number of distinct values in any quantization group of W."""
    W = W.T if columnwise else W
    groups = W.reshape(W.shape[0], -1, group_size).reshape(-1, group_size)
    return max(len(torch.unique(g)) for g in groups)


# ---------------------------------------------------------------- shared behaviour

@pytest.mark.parametrize("args", [
    make_args(method="rtn", high_bits=2),
    make_args(method="bin"),
    make_args(method="lq", high_bits=2),
], ids=["rtn", "bin", "lq"])
def test_weights_modified_in_place_with_same_shape_and_dtype(model, args):
    before = lora_weights(model)
    quantize_adapter(model, args)
    after = lora_weights(model)

    for name in LAYERS:
        for w0, w1 in zip(before[name], after[name]):
            assert w1.shape == w0.shape
            assert w1.dtype == w0.dtype
            assert w1.is_contiguous()
            assert not torch.equal(w0, w1), f"{name} was not quantized"


def test_state_dict_keys_unchanged(model):
    keys = set(get_peft_model_state_dict(model))
    quantize_adapter(model, make_args(method="rtn", high_bits=2))
    assert set(get_peft_model_state_dict(model)) == keys


def test_model_still_runs(model):
    quantize_adapter(model, make_args(method="lq", high_bits=2))
    out = model(torch.randn(4, D_IN))
    assert out.shape == (4, D_OUT) and torch.isfinite(out).all()


# ---------------------------------------------------------------- rtn / bin

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_rtn_avg_bits_and_levels(model, bits):
    avg = quantize_adapter(model, make_args(method="rtn", high_bits=bits))
    assert avg == bits

    for B, A in lora_weights(model).values():
        assert n_distinct_per_group(B, G, columnwise=True) <= 2 ** bits
        assert n_distinct_per_group(A, G, columnwise=False) <= 2 ** bits


def test_bin_avg_bits_and_levels(model):
    avg = quantize_adapter(model, make_args(method="bin"))
    assert avg == 1

    for B, A in lora_weights(model).values():
        # binary: each group is {-s, +s}
        assert n_distinct_per_group(B, G, columnwise=True) <= 2
        assert n_distinct_per_group(A, G, columnwise=False) <= 2


def test_rtn_more_bits_lower_error(model):
    """ΔW error should shrink as bits grow."""
    errs = []
    for bits in [2, 4, 8]:
        m = get_peft_model(Tiny(), LoraConfig(r=R, lora_alpha=16, target_modules=LAYERS, init_lora_weights=False))
        m.load_state_dict(model.state_dict())
        before = lora_weights(m)
        quantize_adapter(m, make_args(method="rtn", high_bits=bits))
        after = lora_weights(m)
        errs.append(sum((before[n][0] @ before[n][1] - after[n][0] @ after[n][1]).norm() for n in LAYERS))
    assert errs[0] > errs[1] > errs[2]


# ---------------------------------------------------------------- lq

def expected_lq_bits(model, rho, high_bits):
    """Average bits from the paper's split, computed independently of quantize_adapter."""
    total_bits = total_params = 0
    for B, A in lora_weights(model).values():
        h = compute_h(rho, svd_split(B, A)[2])
        total_bits += (h * high_bits + (R - h)) * (D_OUT + D_IN)
        total_params += R * (D_OUT + D_IN)
    return total_bits / total_params


@pytest.mark.parametrize("rho", [0.3, 0.7, 0.95])
def test_lq_avg_bits_matches_split(model, rho):
    expected = expected_lq_bits(model, rho, high_bits=3)
    avg = quantize_adapter(model, make_args(method="lq", high_bits=3, ratio=rho))
    assert avg == pytest.approx(expected)
    assert 1 <= avg <= 3


def test_lq_higher_ratio_more_bits(model):
    low = expected_lq_bits(model, 0.3, high_bits=3)
    high = expected_lq_bits(model, 0.95, high_bits=3)
    assert low <= high


def test_lq_with_opt_runs_under_no_grad(model):
    # quantize_adapter wraps the loop in torch.no_grad(); STE must still backprop
    avg = quantize_adapter(model, make_args(method="lq", high_bits=2, opt=True, opt_steps=5))
    assert 1 <= avg <= 2


def test_lq_ignores_opt_steps_without_opt(model):
    """opt=False must not run STE even if opt_steps is set: result equals opt_steps=0."""
    m2 = get_peft_model(Tiny(), LoraConfig(r=R, lora_alpha=16, target_modules=LAYERS, init_lora_weights=False))
    m2.load_state_dict(model.state_dict())

    quantize_adapter(model, make_args(method="lq", high_bits=2, opt=False, opt_steps=50))
    quantize_adapter(m2, make_args(method="lq", high_bits=2, opt=False, opt_steps=0))

    for (B1, A1), (B2, A2) in zip(lora_weights(model).values(), lora_weights(m2).values()):
        assert torch.equal(B1, B2) and torch.equal(A1, A2)
