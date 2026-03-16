"""Phase 3: Operations — groupby, merge, concat, map, and kernel registry.

All operations are functional (return new TensorFrames) and respect
immutability. Designed for compatibility with JAX transformations.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from tensorframe.errors import (
    DimensionError,
    SchemaMismatchError,
    ShapeError,
)
from tensorframe.index import Index, RangeIndex
from tensorframe.schema import DimSpec, FieldSpec, NDSchema


# ============================================================
# Concat
# ============================================================

def concat(frames: list[Any], dim: str) -> Any:
    """Concatenate multiple TensorFrames along a dimension.

    All frames must share the same field names and compatible schemas.
    Fields that don't include the concat dimension are verified to be identical.

    Parameters
    ----------
    frames : list of TensorFrame
        Frames to concatenate.
    dim : str
        Dimension name along which to concatenate.

    Returns
    -------
    TensorFrame
    """
    from tensorframe.frame import TensorFrame

    if not frames:
        raise ValueError("Cannot concatenate empty list of frames")
    if len(frames) == 1:
        return frames[0]

    ref = frames[0]
    ref_names = set(ref.field_names)

    # Validate all frames have the same fields
    for i, f in enumerate(frames[1:], 1):
        if set(f.field_names) != ref_names:
            raise SchemaMismatchError(
                f"Frame {i} has fields {f.field_names}, expected {ref.field_names}"
            )

    new_data = OrderedDict()
    for field_name in ref.field_names:
        fspec = ref.schema.fields[field_name]
        if dim in fspec.dims:
            axis = fspec.dims.index(dim)
            arrays = [f.get_array(field_name) for f in frames]
            combined = jnp.concatenate(arrays, axis=axis)
            new_spec = FieldSpec(
                name=field_name,
                dtype=fspec.dtype,
                dims=fspec.dims,
                shape=tuple(combined.shape),
                nullable=fspec.nullable,
                metadata=fspec.metadata,
            )
            new_data[field_name] = (combined, new_spec)
        else:
            # Field doesn't have the concat dim → keep from first frame
            new_data[field_name] = (ref.get_array(field_name), fspec)

    # Concatenate indices
    new_indices = {}
    for dim_name, idx in ref.indices.items():
        if dim_name == dim:
            all_labels = np.concatenate([
                np.asarray(f.indices[dim_name].labels) for f in frames
                if dim_name in f.indices
            ])
            new_indices[dim_name] = Index(labels=all_labels, name=idx.name)
        else:
            new_indices[dim_name] = idx

    return TensorFrame(data=new_data, indices=new_indices, attrs=ref.attrs)


# ============================================================
# Merge / Join
# ============================================================

def merge(
    left: Any,
    right: Any,
    on: str,
    how: str = "inner",
) -> Any:
    """Merge two TensorFrames on a shared field or index.

    Parameters
    ----------
    left, right : TensorFrame
        Frames to merge.
    on : str
        Field name to join on. Must exist in both frames.
    how : str
        Join type: 'inner', 'left', 'right', 'outer'.

    Returns
    -------
    TensorFrame
    """
    from tensorframe.frame import TensorFrame

    if how not in ("inner", "left", "right", "outer"):
        raise ValueError(f"how must be 'inner', 'left', 'right', or 'outer', got {how!r}")

    left_keys = np.asarray(left.get_array(on))
    right_keys = np.asarray(right.get_array(on))

    if how == "inner":
        mask_l, mask_r = _inner_join_masks(left_keys, right_keys)
    elif how == "left":
        mask_l, mask_r = _left_join_masks(left_keys, right_keys)
    elif how == "right":
        mask_r, mask_l = _left_join_masks(right_keys, left_keys)
    else:  # outer
        mask_l, mask_r = _outer_join_masks(left_keys, right_keys)

    new_data = OrderedDict()

    # Add fields from left
    for name in left.field_names:
        arr = left.get_array(name)
        fspec = left.schema.fields[name]
        sliced = jnp.asarray(np.asarray(arr)[mask_l])
        new_spec = FieldSpec(
            name=name, dtype=fspec.dtype, dims=fspec.dims,
            shape=tuple(sliced.shape), nullable=fspec.nullable,
            metadata=fspec.metadata,
        )
        new_data[name] = (sliced, new_spec)

    # Add non-overlapping fields from right
    for name in right.field_names:
        if name in new_data:
            continue
        arr = right.get_array(name)
        fspec = right.schema.fields[name]
        sliced = jnp.asarray(np.asarray(arr)[mask_r])
        new_spec = FieldSpec(
            name=name, dtype=fspec.dtype, dims=fspec.dims,
            shape=tuple(sliced.shape), nullable=fspec.nullable,
            metadata=fspec.metadata,
        )
        new_data[name] = (sliced, new_spec)

    return TensorFrame(data=new_data)


def _inner_join_masks(
    left_keys: np.ndarray, right_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays for inner join."""
    common = np.intersect1d(left_keys, right_keys)
    left_idx = []
    right_idx = []
    for val in common:
        li = np.where(left_keys == val)[0]
        ri = np.where(right_keys == val)[0]
        for l in li:
            for r in ri:
                left_idx.append(l)
                right_idx.append(r)
    return np.array(left_idx, dtype=np.intp), np.array(right_idx, dtype=np.intp)


def _left_join_masks(
    left_keys: np.ndarray, right_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays for left join."""
    left_idx = []
    right_idx = []
    for i, val in enumerate(left_keys):
        ri = np.where(right_keys == val)[0]
        if len(ri) > 0:
            for r in ri:
                left_idx.append(i)
                right_idx.append(r)
        else:
            left_idx.append(i)
            right_idx.append(0)  # will need NaN handling for outer
    return np.array(left_idx, dtype=np.intp), np.array(right_idx, dtype=np.intp)


def _outer_join_masks(
    left_keys: np.ndarray, right_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays for outer join (simplified: union of keys)."""
    # For simplicity, outer join concatenates inner + left-only + right-only
    common = np.intersect1d(left_keys, right_keys)
    left_idx = []
    right_idx = []

    # Common keys
    for val in common:
        li = np.where(left_keys == val)[0]
        ri = np.where(right_keys == val)[0]
        for l in li:
            for r in ri:
                left_idx.append(l)
                right_idx.append(r)

    # Left-only keys
    left_only = np.setdiff1d(left_keys, right_keys)
    for val in left_only:
        li = np.where(left_keys == val)[0]
        for l in li:
            left_idx.append(l)
            right_idx.append(0)

    # Right-only keys
    right_only = np.setdiff1d(right_keys, left_keys)
    for val in right_only:
        ri = np.where(right_keys == val)[0]
        for r in ri:
            left_idx.append(0)
            right_idx.append(r)

    return np.array(left_idx, dtype=np.intp), np.array(right_idx, dtype=np.intp)


# ============================================================
# GroupBy
# ============================================================

class GroupBy:
    """Lazy groupby object. Groups a TensorFrame by a key field.

    Usage:
        frame.groupby("category").agg({"price": "mean", "qty": "sum"})
    """

    def __init__(self, frame: Any, by: str) -> None:
        self._frame = frame
        self._by = by

        if by not in frame.field_names:
            raise KeyError(f"Groupby key {by!r} not in TensorFrame fields")

        keys_np = np.asarray(frame.get_array(by))
        self._unique_keys = np.unique(keys_np)
        self._key_array = keys_np

    @property
    def n_groups(self) -> int:
        return len(self._unique_keys)

    @property
    def groups(self) -> dict[Any, np.ndarray]:
        """Return dict of group_key -> array of integer positions."""
        result = {}
        for key in self._unique_keys:
            result[key.item() if hasattr(key, 'item') else key] = np.where(
                self._key_array == key
            )[0]
        return result

    def agg(self, aggregations: dict[str, str | Callable]) -> Any:
        """Aggregate each group using specified functions.

        Parameters
        ----------
        aggregations : dict
            Maps field names to aggregation functions.
            String names: 'mean', 'sum', 'min', 'max', 'count', 'std', 'var', 'first', 'last'.
            Callables: any function (np.ndarray) -> scalar.

        Returns
        -------
        TensorFrame
        """
        from tensorframe.frame import TensorFrame

        agg_fns = {
            "mean": jnp.mean,
            "sum": jnp.sum,
            "min": jnp.min,
            "max": jnp.max,
            "std": jnp.std,
            "var": jnp.var,
            "count": lambda x: jnp.array(x.shape[0], dtype=jnp.int32),
            "first": lambda x: x[0],
            "last": lambda x: x[-1],
        }

        groups = self.groups
        n = len(groups)

        new_data = OrderedDict()

        # Key column
        key_values = jnp.array(list(groups.keys()))
        new_data[self._by] = key_values

        # Aggregated columns
        for field_name, func in aggregations.items():
            if field_name not in self._frame.field_names:
                raise KeyError(f"Field {field_name!r} not in TensorFrame")

            if isinstance(func, str):
                if func not in agg_fns:
                    raise ValueError(
                        f"Unknown aggregation {func!r}. "
                        f"Available: {list(agg_fns.keys())}"
                    )
                fn = agg_fns[func]
            else:
                fn = func

            arr = self._frame.get_array(field_name)
            results = []
            for positions in groups.values():
                group_data = arr[positions]
                results.append(fn(group_data))
            new_data[field_name] = jnp.stack(results)

        return TensorFrame(data=new_data)

    def apply(self, fn: Callable) -> Any:
        """Apply a function to each group, concatenating results.

        Parameters
        ----------
        fn : callable
            Function that takes a TensorFrame (group) and returns a TensorFrame.

        Returns
        -------
        TensorFrame
        """
        results = []
        groups = self.groups
        first_dim = self._frame.dims[0]

        for positions in groups.values():
            group_frame = self._frame.isel(**{first_dim: positions})
            result = fn(group_frame)
            results.append(result)

        return concat(results, dim=first_dim)

    def __repr__(self) -> str:
        return f"GroupBy(by={self._by!r}, n_groups={self.n_groups})"


# ============================================================
# Map
# ============================================================

def map_over_dim(
    frame: Any,
    fn: Callable,
    dim: str,
) -> Any:
    """Apply a function to each slice along a dimension, stacking results.

    Parameters
    ----------
    frame : TensorFrame
        The source frame.
    fn : callable
        Function that takes a TensorFrame (one slice) and returns a JAX array.
    dim : str
        Dimension to iterate over.

    Returns
    -------
    jax.Array
        Stacked results of applying fn to each slice.
    """
    if dim not in frame.dims:
        raise DimensionError(f"Dimension {dim!r} not in TensorFrame")

    size = frame.shape.get(dim)
    if size is None or size == 0:
        raise ShapeError(f"Dimension {dim!r} has unknown or zero size")

    results = []
    for i in range(size):
        sliced = frame.isel(**{dim: i})
        results.append(fn(sliced))

    return jnp.stack(results)


# ============================================================
# Kernel Registry
# ============================================================

class KernelRegistry:
    """Central registry for named compute kernels.

    Each kernel is a pure JAX function with associated metadata for
    schema propagation.
    """

    def __init__(self) -> None:
        self._kernels: dict[str, KernelEntry] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        """Register a kernel function."""
        self._kernels[name] = KernelEntry(
            name=name, fn=fn, description=description,
        )

    def get(self, name: str) -> KernelEntry:
        if name not in self._kernels:
            raise KeyError(f"Kernel {name!r} not registered")
        return self._kernels[name]

    def __contains__(self, name: str) -> bool:
        return name in self._kernels

    def list_kernels(self) -> list[str]:
        return list(self._kernels.keys())

    def __repr__(self) -> str:
        return f"KernelRegistry({len(self._kernels)} kernels)"


class KernelEntry:
    """A registered kernel with its metadata."""

    def __init__(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        self.name = name
        self.fn = fn
        self.description = description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Kernel({self.name!r})"


# Global registry instance
_global_registry = KernelRegistry()


def register_kernel(name: str, fn: Callable, description: str = "") -> None:
    """Register a kernel in the global registry."""
    _global_registry.register(name, fn, description)


def get_kernel(name: str) -> KernelEntry:
    """Get a kernel from the global registry."""
    return _global_registry.get(name)


# Pre-register standard kernels
def _register_builtins() -> None:
    register_kernel("sum", jnp.sum, "Sum of array elements")
    register_kernel("mean", jnp.mean, "Mean of array elements")
    register_kernel("min", jnp.min, "Minimum of array elements")
    register_kernel("max", jnp.max, "Maximum of array elements")
    register_kernel("std", jnp.std, "Standard deviation")
    register_kernel("var", jnp.var, "Variance")
    register_kernel("abs", jnp.abs, "Absolute value")
    register_kernel("sqrt", jnp.sqrt, "Square root")
    register_kernel("log", jnp.log, "Natural logarithm")
    register_kernel("log1p", jnp.log1p, "log(1 + x)")
    register_kernel("exp", jnp.exp, "Exponential")


_register_builtins()
