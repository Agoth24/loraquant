"""lq.cli — argument validation and results-path naming (no model / GPU needed)."""
import argparse
import sys

import pytest

from lq.cli import format_results_path, get_args, validate_args

MODEL = "meta-llama/Llama-2-7b-hf"


def make_args(**overrides) -> argparse.Namespace:
    """Namespace with the parser's defaults; override per test."""
    base = dict(
        model=MODEL, method="fp", dataset="gsm8k", adapter_path=".",
        group_size=128, high_bits=None, low_bits=None, ratio=0.7,
        opt=False, opt_steps=100, lr=1e-3, svd="full", limit=None, res_path="outputs",
    )
    return argparse.Namespace(**(base | overrides))


def validate(args):
    validate_args(argparse.ArgumentParser(), args)


# ---------------------------------------------------------------- validate_args

@pytest.mark.parametrize("overrides", [
    dict(method="fp"),
    dict(method="rtn", high_bits=2),
    dict(method="bin"),
    dict(method="lq", high_bits=2),
    dict(method="lq", high_bits=3, ratio=1.0, opt=True, limit=10),
    dict(method="lq", high_bits=2, svd="qr"),
    dict(method="rtn", high_bits=2, svd="full"),
])
def test_valid_args_pass(overrides, tmp_path):
    validate(make_args(adapter_path=str(tmp_path), **overrides))


@pytest.mark.parametrize("overrides, msg", [
    (dict(adapter_path="/does/not/exist"), "--adapter_path"),
    (dict(method="rtn"), "requires --high_bits"),
    (dict(method="lq"), "requires --high_bits"),
    (dict(method="rtn", high_bits=0), "--high_bits must be >= 1"),
    (dict(group_size=0), "--group_size must be > 0"),
    (dict(group_size=-128), "--group_size must be > 0"),
    (dict(method="lq", high_bits=2, ratio=0.0), "--ratio must be in (0, 1]"),
    (dict(method="lq", high_bits=2, ratio=1.5), "--ratio must be in (0, 1]"),
    (dict(method="bin", opt=True), "--opt only applies to --method lq"),
    (dict(method="lq", high_bits=2, opt=True, opt_steps=0), "--opt_steps must be > 0"),
    (dict(method="lq", high_bits=2, opt=True, lr=0.0), "--lr must be > 0"),
    (dict(limit=0), "--limit must be > 0"),
    (dict(method="rtn", high_bits=2, svd="qr"), "--svd qr only applies to --method lq"),
    (dict(method="bin", svd="qr"), "--svd qr only applies to --method lq"),
], ids=lambda x: x if isinstance(x, str) else None)
def test_invalid_args_exit_with_message(overrides, msg, tmp_path, capsys):
    args = make_args(**({"adapter_path": str(tmp_path)} | overrides))
    with pytest.raises(SystemExit) as exc:
        validate(args)
    assert exc.value.code == 2  # argparse's usage-error exit code
    assert msg in capsys.readouterr().err


def test_ratio_ignored_for_non_lq(tmp_path):
    # --ratio is only checked when it is used
    validate(make_args(adapter_path=str(tmp_path), method="rtn", high_bits=2, ratio=5.0))


# ---------------------------------------------------------------- get_args

def test_get_args_parses_and_validates(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "lq", "--model", MODEL, "--adapter_path", str(tmp_path),
        "--method", "lq", "--high_bits", "2", "--opt",
    ])
    args = get_args()
    assert (args.method, args.high_bits, args.opt) == ("lq", 2, True)
    assert args.group_size == 128 and args.ratio == 0.7  # defaults


def test_get_args_svd_default_and_qr(monkeypatch, tmp_path):
    base = ["lq", "--model", MODEL, "--adapter_path", str(tmp_path), "--method", "lq", "--high_bits", "2"]
    monkeypatch.setattr(sys, "argv", base)
    assert get_args().svd == "full"
    monkeypatch.setattr(sys, "argv", base + ["--svd", "qr"])
    assert get_args().svd == "qr"


def test_get_args_rejects_unknown_svd(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "lq", "--model", MODEL, "--adapter_path", str(tmp_path),
        "--method", "lq", "--high_bits", "2", "--svd", "gesvd",
    ])
    with pytest.raises(SystemExit):
        get_args()


def test_get_args_rejects_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "lq", "--model", MODEL, "--adapter_path", str(tmp_path), "--method", "rtn",
    ])
    with pytest.raises(SystemExit):
        get_args()


# ---------------------------------------------------------------- format_results_path

ROOT = "outputs/llama-2-7b-hf/gsm8k"


@pytest.mark.parametrize("overrides, expected", [
    (dict(method="fp"), "fp/fp.json"),
    (dict(method="rtn", high_bits=2), "rtn/rtn_hi2_grp128.json"),
    (dict(method="rtn", high_bits=2, low_bits=1), "rtn/rtn_hi2_lo1_grp128.json"),
    (dict(method="bin", low_bits=1), "bin/bin_grp128.json"),
    (dict(method="lq", high_bits=2, low_bits=1, ratio=0.8), "lq/lq_hi2_grp128_rho0.8.json"),
    (dict(method="lq", high_bits=2, ratio=0.8, opt=True), "lq/lq_hi2_grp128_rho0.8_opt100_lr0.001.json"),
    (dict(method="lq", high_bits=2, ratio=0.8, limit=50), "lq/lq_hi2_grp128_rho0.8_lim50.json"),
    (dict(method="lq", high_bits=2, ratio=0.8, svd="qr"), "lq/lq_hi2_grp128_rho0.8_svdqr.json"),
    (dict(method="lq", high_bits=2, ratio=0.8, svd="full"), "lq/lq_hi2_grp128_rho0.8.json"),
])
def test_results_path(overrides, expected):
    assert str(format_results_path(make_args(**overrides))) == f"{ROOT}/{expected}"


def test_opt_and_no_opt_runs_do_not_collide():
    no_opt = format_results_path(make_args(method="lq", high_bits=2))
    opt = format_results_path(make_args(method="lq", high_bits=2, opt=True))
    assert no_opt != opt


def test_full_and_qr_svd_runs_do_not_collide():
    full = format_results_path(make_args(method="lq", high_bits=2, svd="full"))
    qr = format_results_path(make_args(method="lq", high_bits=2, svd="qr"))
    assert full != qr


def test_opt_settings_change_path():
    a = format_results_path(make_args(method="lq", high_bits=2, opt=True, opt_steps=100))
    b = format_results_path(make_args(method="lq", high_bits=2, opt=True, opt_steps=300))
    c = format_results_path(make_args(method="lq", high_bits=2, opt=True, lr=1e-2))
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("method", ["rtn", "bin"])
def test_unused_settings_not_in_baseline_names(method):
    # ratio and opt settings don't affect rtn/bin, so they must not split their results
    a = format_results_path(make_args(method=method, high_bits=2, ratio=0.5))
    b = format_results_path(make_args(method=method, high_bits=2, ratio=0.9))
    assert a == b


def test_res_path_and_model_dir():
    p = format_results_path(make_args(method="fp", res_path="/tmp/runs"))
    assert p.parts[:3] == ("/", "tmp", "runs")
    assert p.parent.parent.parent.name == "llama-2-7b-hf"
