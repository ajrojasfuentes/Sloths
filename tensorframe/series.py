"""TensorSeries: single-field specialization of TensorFrame."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tensorframe.frame import TensorFrame
from tensorframe.index import Index
from tensorframe.schema import FieldSpec


class TensorSeries(TensorFrame):
    """A TensorFrame with exactly one field.

    Provides convenience accessors to the underlying data without
    needing to specify the field name.
    """

    def __init__(
        self,
        data: dict[str, jnp.ndarray | tuple[jnp.ndarray, FieldSpec]],
        indices: dict[str, Index] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(data=data, indices=indices, attrs=attrs)
        if self.num_fields != 1:
            raise ValueError(
                f"TensorSeries must have exactly 1 field, got {self.num_fields}"
            )

    @property
    def name(self) -> str:
        return self.field_names[0]

    @property
    def values(self) -> jnp.ndarray:
        """The underlying JAX array."""
        return self._data[self.name]

    @property
    def dtype(self) -> Any:
        return self._schema.fields[self.name].dtype

    def to_jax(self) -> jnp.ndarray:
        return self.values

    def __repr__(self) -> str:
        arr = self.values
        return f"TensorSeries(name={self.name!r}, dtype={self.dtype}, shape={arr.shape})"


# --- JAX Pytree registration ---

def _series_flatten(series: TensorSeries) -> tuple[list[jnp.ndarray], Any]:
    leaves = [series._data[name] for name in series._schema.field_names]
    aux = (series._schema, series._indices, series._dim_order, series._attrs)
    return leaves, aux


def _series_unflatten(aux: Any, leaves: list[jnp.ndarray]) -> TensorSeries:
    schema, indices, dim_order, attrs = aux
    frame = object.__new__(TensorSeries)
    data = OrderedDict()
    for i, name in enumerate(schema.field_names):
        data[name] = leaves[i]
    object.__setattr__(frame, "_schema", schema)
    object.__setattr__(frame, "_data", data)
    object.__setattr__(frame, "_indices", indices)
    object.__setattr__(frame, "_dim_order", dim_order)
    object.__setattr__(frame, "_attrs", attrs)
    return frame


jax.tree_util.register_pytree_node(TensorSeries, _series_flatten, _series_unflatten)
