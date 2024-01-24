import logging
from pathlib import Path

from omegaconf import OmegaConf

from nn_core.console_logging import NNRichHandler

import ccmm  # NOQA
from ccmm.matching.weight_matching import LayerIterationOrder  # NOQA

# Required workaround because PyTorch Lightning configures the logging on import,
# thus the logging configuration defined in the __init__.py must be called before
# the lightning import otherwise it has no effect.
# See https://github.com/PyTorchLightning/pytorch-lightning/issues/1503
lightning_logger = logging.getLogger("pytorch_lightning")
# Remove all handlers associated with the lightning logger.
for handler in lightning_logger.handlers[:]:
    lightning_logger.removeHandler(handler)
lightning_logger.propagate = True


def decode_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


OmegaConf.register_new_resolver("path", lambda path: decode_path(path))


def enum_resolver(enum_class: str, enum_member: str):  # NOQA
    enum_class = eval(enum_class)  # NOQA
    return enum_class[enum_member]


OmegaConf.register_new_resolver("enum", enum_resolver)

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        NNRichHandler(
            rich_tracebacks=True,
            show_level=True,
            show_path=True,
            show_time=True,
            omit_repeated_times=True,
        )
    ],
)

try:
    from ._version import __version__ as __version__
except ImportError:
    import sys

    print(
        "Project not installed in the current env, activate the correct env or install it with:\n\tpip install -e .",
        file=sys.stderr,
    )
    __version__ = "unknown"
