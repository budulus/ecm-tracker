"""Small atomic-write primitives for settings, sessions, projects, and plugin exports."""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager

import numpy as np


@contextmanager
def atomic_open(path, mode="w", *, encoding="utf-8", newline=None):
    """Yield a temporary file and replace ``path`` after a successful flush and fsync."""
    path = os.fspath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}-", dir=directory)
    kwargs = {} if "b" in mode else {"encoding": encoding, "newline": newline}
    try:
        with os.fdopen(fd, mode, **kwargs) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def atomic_save_npy(path, array) -> None:
    """Atomically save one numpy array without np.save rewriting the destination suffix."""
    with atomic_open(path, "wb") as f:
        np.save(f, array)
