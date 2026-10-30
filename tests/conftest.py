import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def make_lora(m: int, r: int, n: int, scale: float = 0.02):
    """Random LoRA factors B (m, r), A (r, n) with LoRA-like magnitudes."""
    B = torch.randn(m, r) * scale
    A = torch.randn(r, n) * scale
    return B, A
