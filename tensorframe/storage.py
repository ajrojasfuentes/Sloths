"""Phase 2: Storage Layer — save/open TensorFrames to/from Zarr v3 with TensorStore.

Provides three-tier storage:
  - Level 1 (Hot): JAX arrays in device memory
  - Level 2 (Warm): TensorStore virtual views with async I/O and caching
  - Level 3 (Cold): Zarr v3 persistent chunked/compressed storage
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any

import jax.numpy as jnp
import numpy as np
import zarr
import tensorstore as ts

from tensorframe.errors import PersistenceError, StorageError
from tensorframe.index import Index, RangeIndex
from tensorframe.ndtype import NDType
from tensorframe.schema import DimSpec, FieldSpec, NDSchema


# --- Zarr dtype mapping ---

_NDTYPE_TO_ZARR_DTYPE = {
    "bool_": "|b1",
    "int8": "<i1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "uint8": "<u1",
    "uint16": "<u2",
    "uint32": "<u4",
    "uint64": "<u8",
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "bfloat16": "<f2",  # stored as float16, converted at load time
    "complex64": "<c8",
    "complex128": "<c16",
}

_ZARR_DTYPE_TO_JAX = {
    "|b1": jnp.bool_,
    "<i1": jnp.int8,
    "<i2": jnp.int16,
    "<i4": jnp.int32,
    "<i8": jnp.int64,
    "<u1": jnp.uint8,
    "<u2": jnp.uint16,
    "<u4": jnp.uint32,
    "<u8": jnp.uint64,
    "<f2": jnp.float16,
    "<f4": jnp.float32,
    "<f8": jnp.float64,
    "<c8": jnp.complex64,
    "<c16": jnp.complex128,
}


def _resolve_zarr_dtype(field_spec: FieldSpec) -> str:
    """Map NDType to a zarr-compatible dtype string."""
    base_name = field_spec.dtype.name
    # For composite types (tensor, nullable, etc.), use the jax_dtype
    if base_name in _NDTYPE_TO_ZARR_DTYPE:
        return _NDTYPE_TO_ZARR_DTYPE[base_name]
    if field_spec.dtype.jax_dtype is not None:
        name = str(jnp.dtype(field_spec.dtype.jax_dtype))
        for k, v in _ZARR_DTYPE_TO_JAX.items():
            if str(jnp.dtype(v)) == name:
                return k
    return "<f4"  # default float32


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose sensible default chunk sizes."""
    if not shape:
        return ()
    chunks = []
    for i, s in enumerate(shape):
        if i == 0:
            # First dim (batch): chunk at most 256 elements
            chunks.append(min(s, 256))
        else:
            # Inner dims: keep whole
            chunks.append(s)
    return tuple(chunks)


# ============================================================
# SAVE: TensorFrame → Zarr v3
# ============================================================

def save(
    frame: Any,  # TensorFrame, imported lazily to avoid circular import
    path: str,
    chunk_config: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Save a TensorFrame to Zarr v3 format on disk.

    Parameters
    ----------
    frame : TensorFrame
        The frame to persist.
    path : str
        Path for the .zarr directory.
    chunk_config : dict, optional
        Per-field chunking config. Keys are field names, values are dicts
        with optional 'chunks' (tuple) and 'codecs' (list of codec names).
    """
    chunk_config = chunk_config or {}

    try:
        store = zarr.storage.LocalStore(path)
        root = zarr.open_group(store=store, mode="w")

        # Save schema as JSON attribute
        root.attrs["tensorframe_schema"] = frame.schema.to_dict()
        root.attrs["tensorframe_version"] = "0.1.0"

        # Save each field as a zarr array
        for field_name in frame.field_names:
            arr_data = np.asarray(frame.get_array(field_name))
            fspec = frame.schema.fields[field_name]

            # Determine chunks
            fc = chunk_config.get(field_name, {})
            chunks = fc.get("chunks", _default_chunks(arr_data.shape))

            zarr_dtype = _resolve_zarr_dtype(fspec)

            root.create_array(
                field_name,
                data=arr_data.astype(zarr_dtype),
                chunks=chunks,
            )

        # Save indices
        indices_group = root.create_group("_indices")
        for dim_name, idx in frame.indices.items():
            idx_data = idx.to_dict()
            if isinstance(idx, RangeIndex):
                indices_group.attrs[dim_name] = idx_data
            else:
                indices_group.create_array(
                    dim_name,
                    data=idx.labels,
                    chunks=(len(idx.labels),),
                )
                indices_group.attrs[dim_name] = {
                    "kind": "index",
                    "name": idx.name,
                    "dtype": str(idx.labels.dtype),
                }

    except Exception as e:
        raise PersistenceError(f"Failed to save TensorFrame to {path}: {e}") from e


# ============================================================
# OPEN: Zarr v3 → TensorFrame (lazy or eager)
# ============================================================

def open(path: str, lazy: bool = False) -> Any:
    """Open a TensorFrame from a Zarr v3 store.

    Parameters
    ----------
    path : str
        Path to the .zarr directory.
    lazy : bool
        If True, return a LazyTensorFrame that loads data on demand.
        If False (default), load all data into JAX arrays immediately.

    Returns
    -------
    TensorFrame or LazyTensorFrame
    """
    from tensorframe.frame import TensorFrame

    if not os.path.exists(path):
        raise StorageError(f"Path does not exist: {path}")

    try:
        store = zarr.storage.LocalStore(path)
        root = zarr.open_group(store=store, mode="r")

        schema_dict = dict(root.attrs.get("tensorframe_schema", {}))
        schema = NDSchema.from_dict(schema_dict)

        # Load indices
        indices: dict[str, Index] = {}
        if "_indices" in root:
            idx_group = root["_indices"]
            idx_attrs = dict(idx_group.attrs)
            for dim_name, meta in idx_attrs.items():
                meta = dict(meta)
                if meta.get("kind") == "range_index":
                    indices[dim_name] = RangeIndex.from_dict(meta)
                elif dim_name in idx_group:
                    labels = np.asarray(idx_group[dim_name])
                    indices[dim_name] = Index(
                        labels=labels,
                        name=meta.get("name"),
                    )

        if lazy:
            return LazyTensorFrame(path=path, schema=schema, indices=indices)

        # Eager load: read all fields into JAX arrays
        data = OrderedDict()
        for field_name, fspec in schema.fields.items():
            arr_np = np.asarray(root[field_name])
            jax_dtype = fspec.dtype.jax_dtype
            if jax_dtype is not None:
                arr_jax = jnp.array(arr_np, dtype=jax_dtype)
            else:
                arr_jax = jnp.array(arr_np)
            data[field_name] = (arr_jax, fspec)

        return TensorFrame(data=data, indices=indices)

    except (PersistenceError, StorageError):
        raise
    except Exception as e:
        raise StorageError(f"Failed to open TensorFrame from {path}: {e}") from e


# ============================================================
# TensorStore-backed lazy loading
# ============================================================

class LazyTensorFrame:
    """A TensorFrame where fields are loaded on demand via TensorStore.

    Fields start in 'cold' state and are materialized to JAX arrays
    when accessed. Provides the same read API as TensorFrame.
    """

    def __init__(
        self,
        path: str,
        schema: NDSchema,
        indices: dict[str, Index],
    ) -> None:
        self._path = os.path.abspath(path)
        self._schema = schema
        self._indices = indices
        self._cache: dict[str, jnp.ndarray] = {}

    @property
    def schema(self) -> NDSchema:
        return self._schema

    @property
    def field_names(self) -> list[str]:
        return self._schema.field_names

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self._schema.dims.keys())

    @property
    def indices(self) -> dict[str, Index]:
        return dict(self._indices)

    @property
    def num_fields(self) -> int:
        return self._schema.num_fields

    @property
    def shape(self) -> dict[str, int | None]:
        return {d.name: d.size for d in self._schema.dims.values()}

    def _materialize_field(self, field_name: str) -> jnp.ndarray:
        """Load a single field from disk into a JAX array using TensorStore."""
        if field_name in self._cache:
            return self._cache[field_name]

        fspec = self._schema.fields[field_name]
        zarr_dtype = _resolve_zarr_dtype(fspec)

        field_path = os.path.join(self._path, field_name)

        spec = {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": field_path,
            },
        }

        try:
            dataset = ts.open(spec).result()
            arr_np = dataset.read().result()
            jax_dtype = fspec.dtype.jax_dtype
            if jax_dtype is not None:
                arr_jax = jnp.array(arr_np, dtype=jax_dtype)
            else:
                arr_jax = jnp.array(arr_np)
            self._cache[field_name] = arr_jax
            return arr_jax
        except Exception as e:
            from tensorframe.errors import MaterializationError
            raise MaterializationError(
                f"Failed to materialize field {field_name!r} from {field_path}: {e}"
            ) from e

    def get_array(self, field_name: str) -> jnp.ndarray:
        if field_name not in self._schema.fields:
            raise KeyError(f"Field {field_name!r} not in LazyTensorFrame")
        return self._materialize_field(field_name)

    def __getitem__(self, key: str) -> jnp.ndarray:
        return self.get_array(key)

    def compute(self) -> Any:
        """Materialize all fields and return a fully-loaded TensorFrame."""
        from tensorframe.frame import TensorFrame

        data = OrderedDict()
        for field_name, fspec in self._schema.fields.items():
            arr = self._materialize_field(field_name)
            data[field_name] = (arr, fspec)
        return TensorFrame(data=data, indices=self._indices)

    def is_cached(self, field_name: str) -> bool:
        """Check if a field has already been materialized."""
        return field_name in self._cache

    def evict(self, field_name: str) -> None:
        """Remove a field from the in-memory cache."""
        self._cache.pop(field_name, None)

    def evict_all(self) -> None:
        """Remove all fields from cache."""
        self._cache.clear()

    def __len__(self) -> int:
        first_field = self._schema.field_names[0]
        fspec = self._schema.fields[first_field]
        if fspec.shape:
            return fspec.shape[0] if fspec.shape[0] is not None else 0
        return 0

    def __repr__(self) -> str:
        cached = [f for f in self.field_names if f in self._cache]
        return (
            f"LazyTensorFrame(path={self._path!r}, fields={self.num_fields}, "
            f"cached={len(cached)}/{self.num_fields})"
        )

    def __contains__(self, key: str) -> bool:
        return key in self._schema.fields
