import importlib.metadata

from .Tokenizer import Tokenizer

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Tokenizer"]
