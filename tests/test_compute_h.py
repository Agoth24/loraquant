"""compute_h — paper §3.1 (variance ratio after Eq. 4), Alg. 1 line 4."""
import pytest
import torch

from lq.method_utils import compute_h


S_EXAMPLE = torch.tensor([3.0, 2.0, 1.0])  # energies 9, 4, 1 -> ratios 9/14, 13/14, 1


@pytest.mark.parametrize(
    "rho, expected",
    [(0.1, 1), (0.6, 1), (0.9, 2), (0.95, 3)],
)
def test_hand_computed(rho, expected):
    assert compute_h(rho, S_EXAMPLE) == expected


def test_threshold_is_inclusive():
    # ratios are exactly 0.25, 0.5, 0.75, 1.0 -> ">= rho" means rho=0.5 gives h=2
    S = torch.ones(4)
    assert compute_h(0.5, S) == 2


def test_uses_squared_singular_values():
    # S=[3,1]: raw ratio 3/4=0.75, squared ratio 9/10=0.9
    # rho=0.85 -> h=1 with squares, h=2 with raw values
    assert compute_h(0.85, torch.tensor([3.0, 1.0])) == 1


def test_rho_one_returns_full_rank():
    # paper: rho in (0, 1]; rho=1 must keep all r components
    S, _ = torch.rand(16).sort(descending=True)
    assert compute_h(1.0, S) == 16


def test_returns_python_int():
    assert type(compute_h(0.5, S_EXAMPLE)) is int


def test_monotone_in_rho():
    S, _ = torch.rand(16).sort(descending=True)
    hs = [compute_h(rho, S) for rho in torch.linspace(0.05, 1.0, 20).tolist()]
    assert hs == sorted(hs)
    assert all(1 <= h <= 16 for h in hs)
