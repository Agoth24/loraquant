from transformers.utils import logging as hf_logging
import logging
import time
import sys


# configure root logging as "HH:MM:SS | message" and silence library chatter
def setup_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # silence library chatter: only warnings and errors from transformers / lm_eval
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("lm_eval").setLevel(logging.WARNING)


# elapsed time since `start` as e.g. "4m56s"
def elapsed(start: float) -> str:
    m, s = divmod(int(time.perf_counter() - start), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
