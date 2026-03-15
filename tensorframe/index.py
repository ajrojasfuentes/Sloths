"""Index implementations for labeled dimensions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Index:
    """Labeled index for a dimension, backed by a NumPy array.

    Index labels live on the host (NumPy, not JAX) because they are metadata
    used for alignment and selection, not for accelerated computation.
    """

    labels: np.ndarray
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.labels, np.ndarray):
            object.__setattr__(self, "labels", np.asarray(self.labels))
        if self.labels.ndim != 1:
            raise ValueError(f"Index labels must be 1D, got {self.labels.ndim}D")

    def __len__(self) -> int:
        return len(self.labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self.name == other.name and np.array_equal(self.labels, other.labels)

    def __hash__(self) -> int:
        return hash((self.name, self.labels.tobytes()))

    def __contains__(self, label: Any) -> bool:
        return bool(np.isin(label, self.labels))

    def __repr__(self) -> str:
        n = len(self.labels)
        name_str = f", name={self.name!r}" if self.name else ""
        if n <= 6:
            vals = list(self.labels)
        else:
            vals = list(self.labels[:3]) + ["..."] + list(self.labels[-3:])
        return f"Index({vals}{name_str})"

    def get_loc(self, label: Any) -> int:
        """Return the integer position of a label."""
        matches = np.where(self.labels == label)[0]
        if len(matches) == 0:
            from tensorframe.errors import IndexLabelError
            raise IndexLabelError(f"Label {label!r} not found in index")
        return int(matches[0])

    def get_locs(self, labels: list | np.ndarray) -> np.ndarray:
        """Return integer positions for multiple labels."""
        labels_arr = np.asarray(labels)
        positions = []
        for lab in labels_arr:
            positions.append(self.get_loc(lab))
        return np.array(positions, dtype=np.intp)

    def slice_locs(self, start: Any = None, stop: Any = None) -> tuple[int, int]:
        """Return (start_pos, stop_pos) for a label-based slice (inclusive stop)."""
        s = 0 if start is None else self.get_loc(start)
        e = len(self) if stop is None else self.get_loc(stop) + 1
        return s, e

    def rename(self, name: str) -> Index:
        return Index(labels=self.labels, name=name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "index",
            "name": self.name,
            "labels": self.labels.tolist(),
            "dtype": str(self.labels.dtype),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Index:
        if d.get("kind") == "range_index":
            return RangeIndex.from_dict(d)
        return Index(
            labels=np.array(d["labels"], dtype=d.get("dtype", None)),
            name=d.get("name"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(s: str) -> Index:
        return Index.from_dict(json.loads(s))


@dataclass(frozen=True)
class RangeIndex(Index):
    """Memory-efficient index for integer ranges (like pandas.RangeIndex).

    Stores only start/stop/step instead of materializing all labels.
    """

    start: int = 0
    stop: int = 0
    step: int = 1

    def __init__(
        self,
        stop: int | None = None,
        *,
        start: int = 0,
        step: int = 1,
        name: str | None = None,
    ) -> None:
        if stop is None:
            stop = 0
        labels = np.arange(start, stop, step)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step", step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "range_index",
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "step": self.step,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RangeIndex:
        return RangeIndex(
            start=d.get("start", 0),
            stop=d.get("stop", 0),
            step=d.get("step", 1),
            name=d.get("name"),
        )

    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step}{name_str})"
