"""Phase 4: ML Pipeline — normalize, split, encode_categorical, iter_batches, to_jax_arrays.

Provides data cleaning, normalization, encoding, and batching utilities
designed for ML/DL workflows built on JAX.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generator

import jax
import jax.numpy as jnp
import numpy as np

from tensorframe.errors import DimensionError, ShapeError
from tensorframe.index import Index, RangeIndex
from tensorframe.schema import FieldSpec


# ============================================================
# Null handling
# ============================================================

def dropna(frame: Any, dim: str | None = None, fields: list[str] | None = None) -> Any:
    """Drop elements with NaN values along a dimension.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    dim : str, optional
        Dimension to drop along. Defaults to first dimension.
    fields : list of str, optional
        Fields to check for NaN. Defaults to all numeric fields.

    Returns
    -------
    TensorFrame
    """
    if dim is None:
        dim = frame.dims[0]

    target_fields = fields if fields is not None else frame.field_names

    # Build a combined mask: True where ALL target fields are non-NaN
    mask = None
    for name in target_fields:
        arr = frame.get_array(name)
        if not jnp.issubdtype(arr.dtype, jnp.floating):
            continue
        # Reduce all axes except the target dim
        fspec = frame.schema.fields[name]
        if dim not in fspec.dims:
            continue
        axis = fspec.dims.index(dim)
        # isnan along all other axes
        field_valid = ~jnp.isnan(arr)
        # Reduce to 1D along `dim`
        reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
        if reduce_axes:
            field_valid = jnp.all(field_valid, axis=reduce_axes)
        if mask is None:
            mask = field_valid
        else:
            mask = mask & field_valid

    if mask is None:
        return frame

    return frame.where(mask, dim=dim)


def fillna(frame: Any, fill_values: dict[str, Any]) -> Any:
    """Replace NaN values in specified fields.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    fill_values : dict
        Maps field names to fill values (scalars).

    Returns
    -------
    TensorFrame
    """
    new_data = OrderedDict()
    for name in frame.field_names:
        arr = frame.get_array(name)
        fspec = frame.schema.fields[name]
        if name in fill_values:
            fill_val = fill_values[name]
            filled = jnp.where(jnp.isnan(arr), fill_val, arr)
            new_data[name] = (filled, fspec)
        else:
            new_data[name] = (arr, fspec)

    from tensorframe.frame import TensorFrame
    return TensorFrame(data=new_data, indices=frame.indices, attrs=frame.attrs)


# ============================================================
# Normalization
# ============================================================

def normalize(
    frame: Any,
    fields: list[str],
    method: str = "zscore",
    dim: str | tuple[str, ...] | None = None,
    return_params: bool = False,
) -> Any:
    """Normalize selected fields.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    fields : list of str
        Fields to normalize.
    method : str
        'zscore' (mean=0, std=1) or 'minmax' (min=0, max=1).
    dim : str or tuple of str, optional
        Dimensions to compute statistics over. Defaults to first dimension.
    return_params : bool
        If True, return (frame, params_dict) where params_dict contains
        the statistics used for normalization.

    Returns
    -------
    TensorFrame or (TensorFrame, dict)
    """
    if method not in ("zscore", "minmax"):
        raise ValueError(f"method must be 'zscore' or 'minmax', got {method!r}")

    new_data = OrderedDict()
    params = {}

    for name in frame.field_names:
        arr = frame.get_array(name)
        fspec = frame.schema.fields[name]

        if name not in fields:
            new_data[name] = (arr, fspec)
            continue

        # Determine axes to reduce
        if dim is None:
            axes = (0,)
        elif isinstance(dim, str):
            if dim not in fspec.dims:
                new_data[name] = (arr, fspec)
                continue
            axes = (fspec.dims.index(dim),)
        else:
            axes = tuple(fspec.dims.index(d) for d in dim if d in fspec.dims)
            if not axes:
                new_data[name] = (arr, fspec)
                continue

        if method == "zscore":
            mean = jnp.mean(arr, axis=axes, keepdims=True)
            std = jnp.std(arr, axis=axes, keepdims=True)
            std = jnp.where(std == 0, 1.0, std)  # avoid division by zero
            normalized = (arr - mean) / std
            params[name] = {"mean": mean, "std": std}
        else:  # minmax
            vmin = jnp.min(arr, axis=axes, keepdims=True)
            vmax = jnp.max(arr, axis=axes, keepdims=True)
            denom = vmax - vmin
            denom = jnp.where(denom == 0, 1.0, denom)
            normalized = (arr - vmin) / denom
            params[name] = {"min": vmin, "max": vmax}

        new_data[name] = (normalized, fspec)

    from tensorframe.frame import TensorFrame
    result = TensorFrame(data=new_data, indices=frame.indices, attrs=frame.attrs)

    if return_params:
        return result, params
    return result


# ============================================================
# Categorical encoding
# ============================================================

def encode_categorical(
    frame: Any,
    field_name: str,
    categories: list[Any] | None = None,
) -> Any:
    """Encode a field as integer category codes.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    field_name : str
        Field to encode.
    categories : list, optional
        Explicit category list. If None, derived from unique values.

    Returns
    -------
    TensorFrame
        With the field replaced by integer codes.
    """
    arr_np = np.asarray(frame.get_array(field_name))

    if categories is None:
        categories = sorted(np.unique(arr_np).tolist())

    cat_map = {cat: i for i, cat in enumerate(categories)}
    codes = np.array([cat_map.get(v, -1) for v in arr_np.flat], dtype=np.int32)
    codes = codes.reshape(arr_np.shape)

    from tensorframe.ndtype import categorical as cat_type
    fspec = frame.schema.fields[field_name]
    cat_dt = cat_type([str(c) for c in categories])
    new_spec = FieldSpec(
        name=field_name,
        dtype=cat_dt,
        dims=fspec.dims,
        shape=fspec.shape,
        nullable=fspec.nullable,
        metadata={**fspec.metadata, "categories": categories},
    )

    return frame.with_column(field_name, (jnp.array(codes), new_spec))


def one_hot(
    frame: Any,
    field_name: str,
    num_classes: int | None = None,
) -> Any:
    """One-hot encode a categorical integer field.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    field_name : str
        Integer field to one-hot encode.
    num_classes : int, optional
        Number of classes. Inferred from max value + 1 if not given.

    Returns
    -------
    TensorFrame
        With the field replaced by a one-hot encoded array.
    """
    arr = frame.get_array(field_name)
    if num_classes is None:
        num_classes = int(jnp.max(arr)) + 1

    encoded = jax.nn.one_hot(arr, num_classes)
    fspec = frame.schema.fields[field_name]

    from tensorframe.ndtype import float32 as f32
    oh_dim = f"{field_name}_class"
    new_dims = fspec.dims + (oh_dim,)
    new_spec = FieldSpec(
        name=field_name,
        dtype=f32,
        dims=new_dims,
        shape=tuple(encoded.shape),
        nullable=False,
        metadata=fspec.metadata,
    )
    return frame.with_column(field_name, (encoded, new_spec))


# ============================================================
# Train/Val/Test Split
# ============================================================

def split(
    frame: Any,
    dim: str | None = None,
    ratios: list[float] | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[Any, ...]:
    """Split a TensorFrame into train/val/test (or any number of splits).

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    dim : str, optional
        Dimension to split along. Defaults to first dimension.
    ratios : list of float, optional
        Split ratios. Must sum to 1.0. Default: [0.7, 0.15, 0.15].
    shuffle : bool
        Whether to shuffle before splitting.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of TensorFrame
    """
    if ratios is None:
        ratios = [0.7, 0.15, 0.15]

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    if dim is None:
        dim = frame.dims[0]

    n = frame.shape.get(dim)
    if n is None or n == 0:
        raise ShapeError(f"Cannot split along dimension {dim!r} with size {n}")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    splits = []
    start = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            end = n  # last split gets remaining
        else:
            end = start + int(round(ratio * n))
        split_idx = indices[start:end]
        splits.append(frame.isel(**{dim: split_idx}))
        start = end

    return tuple(splits)


# ============================================================
# to_jax_arrays: extract feature/target tensors for ML
# ============================================================

def to_jax_arrays(
    frame: Any,
    features: list[str],
    target: str | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Extract feature and target arrays for ML training.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    features : list of str
        Field names to stack as feature columns.
    target : str, optional
        Target field name.

    Returns
    -------
    tuple
        (X,) if no target, (X, y) if target specified.
        X has shape (N, num_features) for 1D fields.
    """
    feature_arrays = []
    for name in features:
        arr = frame.get_array(name)
        if arr.ndim == 1:
            arr = arr[:, None]  # reshape (N,) → (N, 1)
        feature_arrays.append(arr)

    X = jnp.concatenate(feature_arrays, axis=-1) if len(feature_arrays) > 1 else feature_arrays[0]

    if target is None:
        return (X,)

    y = frame.get_array(target)
    return X, y


# ============================================================
# Batch iteration
# ============================================================

def iter_batches(
    frame: Any,
    batch_size: int,
    dim: str | None = None,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> Generator[Any, None, None]:
    """Iterate over a TensorFrame in batches.

    Parameters
    ----------
    frame : TensorFrame
        Source frame.
    batch_size : int
        Number of elements per batch.
    dim : str, optional
        Dimension to batch along. Defaults to first dimension.
    shuffle : bool
        Shuffle indices before batching.
    seed : int
        Random seed for shuffle.
    drop_last : bool
        If True, drop the last incomplete batch.

    Yields
    ------
    TensorFrame
        Batches of the specified size.
    """
    if dim is None:
        dim = frame.dims[0]

    n = frame.shape.get(dim)
    if n is None or n == 0:
        return

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        end = start + batch_size
        if end > n and drop_last:
            return
        batch_idx = indices[start:min(end, n)]
        yield frame.isel(**{dim: batch_idx})
