"""lq.utils — elapsed-time formatting and logging setup."""
import logging

import pytest

import lq.utils as utils


@pytest.mark.parametrize("seconds, expected", [
    (0, "0s"),
    (5.9, "5s"),       # truncates, doesn't round
    (59, "59s"),
    (60, "1m00s"),
    (296, "4m56s"),
    (3725, "62m05s"),  # no hours unit
])
def test_elapsed(monkeypatch, seconds, expected):
    monkeypatch.setattr(utils.time, "perf_counter", lambda: 1000.0 + seconds)
    assert utils.elapsed(1000.0) == expected


def test_setup_logging_format():
    utils.setup_logging()
    root = logging.getLogger()
    assert root.level == logging.INFO
    fmt = root.handlers[0].formatter
    assert fmt._fmt == "%(asctime)s | %(message)s"
    assert fmt.datefmt == "%H:%M:%S"
    assert logging.getLogger("lm_eval").level == logging.WARNING
