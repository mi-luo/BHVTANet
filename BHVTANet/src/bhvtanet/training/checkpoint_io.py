"""Atomic checkpoint storage on a local filesystem."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit

import torch
from pytorch_lightning.plugins.io import CheckpointIO


def _local_path(path: str | Path) -> Path:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise TypeError("Checkpoint paths must be strings or pathlib paths.")
    windows_drive = re.match(r"^[A-Za-z]:[\\/]", value) is not None
    if not windows_drive and urlsplit(value).scheme:
        raise ValueError("LocalAtomicCheckpointIO requires a local filesystem path.")
    if "://" in value:
        raise ValueError("Checkpoint URI schemes are not supported; supply a local path.")
    return Path(value).expanduser().absolute()


class LocalAtomicCheckpointIO(CheckpointIO):
    """Write beside the destination, then replace it without a cross-device move."""

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
        storage_options: Any | None = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError("LocalAtomicCheckpointIO does not accept storage_options.")
        destination = _local_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: Any | None = None,
        weights_only: bool | None = None,
    ) -> dict[str, Any]:
        return torch.load(
            _local_path(path),
            map_location="cpu" if map_location is None else map_location,
            weights_only=False if weights_only is None else weights_only,
        )

    def remove_checkpoint(self, path: str | Path) -> None:
        _local_path(path).unlink(missing_ok=True)
