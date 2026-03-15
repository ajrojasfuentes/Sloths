"""Helper functions for constructing TensorFrame fields."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tensorframe.ndtype import NDType, ScalarType, float32
from tensorframe.schema import FieldSpec


def _infer_dtype(arr: jnp.ndarray) -> NDType:
    """Infer an NDType from a JAX array's dtype."""
    from tensorframe import ndtype as ndt

    # Map by dtype name string for reliable matching across JAX/NumPy dtype representations
    name_map = {
        "bool": ndt.bool_,
        "int8": ndt.int8,
        "int16": ndt.int16,
        "int32": ndt.int32,
        "int64": ndt.int64,
        "uint8": ndt.uint8,
        "uint16": ndt.uint16,
        "uint32": ndt.uint32,
        "uint64": ndt.uint64,
        "float16": ndt.float16,
        "float32": ndt.float32,
        "float64": ndt.float64,
        "bfloat16": ndt.bfloat16,
        "complex64": ndt.complex64,
        "complex128": ndt.complex128,
    }
    dtype_name = str(arr.dtype)
    return name_map.get(dtype_name, ndt.float32)


def field(
    data: Any,
    dims: tuple[str, ...] | None = None,
    dtype: NDType | None = None,
    name: str | None = None,
) -> tuple[jnp.ndarray, FieldSpec]:
    """Create a field from data, returning (jax_array, FieldSpec).

    Parameters
    ----------
    data : array-like
        Data to wrap. Converted to jax.Array.
    dims : tuple of str, optional
        Dimension names. If None, auto-generated as ("dim_0", "dim_1", ...).
    dtype : NDType, optional
        Explicit type. If None, inferred from the data.
    name : str, optional
        Field name. Can be set later during TensorFrame construction.
    """
    arr = jnp.asarray(data)
    if dtype is not None and isinstance(dtype, ScalarType) and dtype.jax_dtype is not None:
        arr = arr.astype(dtype.jax_dtype)

    if dims is None:
        dims = tuple(f"dim_{i}" for i in range(arr.ndim))

    if len(dims) != arr.ndim:
        raise ValueError(
            f"dims length ({len(dims)}) must match array ndim ({arr.ndim})"
        )

    inferred_dtype = dtype if dtype is not None else _infer_dtype(arr)
    field_name = name or "__unnamed__"

    spec = FieldSpec(
        name=field_name,
        dtype=inferred_dtype,
        dims=dims,
        shape=tuple(arr.shape),
    )
    return arr, spec


def tensor_field(
    data: Any,
    dims: tuple[str, ...] | None = None,
    name: str | None = None,
) -> tuple[jnp.ndarray, FieldSpec]:
    """Create a tensor field (multi-dimensional per-element data).

    Like field() but uses TensorType for the dtype.
    """
    from tensorframe.ndtype import tensor as tensor_type

    arr = jnp.asarray(data)

    if dims is None:
        dims = tuple(f"dim_{i}" for i in range(arr.ndim))

    if len(dims) != arr.ndim:
        raise ValueError(
            f"dims length ({len(dims)}) must match array ndim ({arr.ndim})"
        )

    inner_dtype = _infer_dtype(arr)
    # For tensor fields, the inner shape is everything after the first dim
    inner_shape = arr.shape[1:] if arr.ndim > 1 else ()
    dt = tensor_type(inner_dtype, inner_shape)
    field_name = name or "__unnamed__"

    spec = FieldSpec(
        name=field_name,
        dtype=dt,
        dims=dims,
        shape=tuple(arr.shape),
    )
    return arr, spec
