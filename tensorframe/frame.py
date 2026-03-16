"""TensorFrame: immutable N-dimensional labeled data container on JAX."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from tensorframe.errors import (
    DimensionError,
    IndexLabelError,
    SchemaValidationError,
    SchemaMismatchError,
    ShapeError,
)
from tensorframe.index import Index, RangeIndex
from tensorframe.ndtype import NDType
from tensorframe.schema import DimSpec, FieldSpec, NDSchema
from tensorframe.construction import field as make_field, _infer_dtype


class TensorFrame:
    """Immutable N-dimensional labeled data container backed by JAX arrays.

    Each field is an independent jax.Array. All fields share at least one
    alignment dimension (the primary axis). The TensorFrame is registered
    as a JAX pytree so it can be used with jit, grad, vmap, etc.
    """

    __slots__ = ("_schema", "_data", "_indices", "_dim_order", "_attrs")

    def __init__(
        self,
        data: dict[str, jnp.ndarray | tuple[jnp.ndarray, FieldSpec]],
        indices: dict[str, Index] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        fields_od: OrderedDict[str, FieldSpec] = OrderedDict()
        data_od: OrderedDict[str, jnp.ndarray] = OrderedDict()
        all_dims: OrderedDict[str, DimSpec] = OrderedDict()

        for name, value in data.items():
            if isinstance(value, tuple) and len(value) == 2:
                arr, spec = value
                arr = jnp.asarray(arr)
                # Fix the name in the spec if it was placeholder
                if spec.name != name:
                    spec = FieldSpec(
                        name=name,
                        dtype=spec.dtype,
                        dims=spec.dims,
                        shape=spec.shape,
                        nullable=spec.nullable,
                        metadata=spec.metadata,
                    )
            else:
                arr = jnp.asarray(value)
                dtype = _infer_dtype(arr)
                dims = tuple(f"dim_{i}" for i in range(arr.ndim))
                spec = FieldSpec(
                    name=name, dtype=dtype, dims=dims, shape=tuple(arr.shape)
                )

            fields_od[name] = spec
            data_od[name] = arr

            for i, dim_name in enumerate(spec.dims):
                size = arr.shape[i]
                if dim_name in all_dims:
                    existing = all_dims[dim_name]
                    if existing.size is not None and existing.size != size:
                        raise ShapeError(
                            f"Dimension {dim_name!r} has conflicting sizes: "
                            f"{existing.size} vs {size} (field {name!r})"
                        )
                else:
                    all_dims[dim_name] = DimSpec(name=dim_name, size=size)

        schema = NDSchema(fields=fields_od, dims=all_dims)

        # Build indices
        resolved_indices: dict[str, Index] = {}
        if indices is not None:
            for dim_name, idx in indices.items():
                if dim_name not in all_dims:
                    raise DimensionError(
                        f"Index for dimension {dim_name!r} but dimension not in schema"
                    )
                expected_size = all_dims[dim_name].size
                if expected_size is not None and len(idx) != expected_size:
                    raise ShapeError(
                        f"Index for {dim_name!r} has length {len(idx)} "
                        f"but dimension size is {expected_size}"
                    )
                resolved_indices[dim_name] = idx

        # Auto-create RangeIndex for dims without explicit index
        for dim_name, dim_spec in all_dims.items():
            if dim_name not in resolved_indices and dim_spec.size is not None:
                resolved_indices[dim_name] = RangeIndex(
                    stop=dim_spec.size, name=dim_name
                )

        dim_order = tuple(all_dims.keys())

        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_data", data_od)
        object.__setattr__(self, "_indices", resolved_indices)
        object.__setattr__(self, "_dim_order", dim_order)
        object.__setattr__(self, "_attrs", attrs or {})

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("TensorFrame is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("TensorFrame is immutable")

    # --- Properties ---

    @property
    def schema(self) -> NDSchema:
        return self._schema

    @property
    def field_names(self) -> list[str]:
        return self._schema.field_names

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dim_order

    @property
    def indices(self) -> dict[str, Index]:
        return dict(self._indices)

    @property
    def shape(self) -> dict[str, int | None]:
        return {d.name: d.size for d in self._schema.dims.values()}

    @property
    def num_fields(self) -> int:
        return self._schema.num_fields

    @property
    def attrs(self) -> dict[str, Any]:
        return dict(self._attrs)

    # --- Field access ---

    def __getitem__(self, key: str | list[str]) -> Any:
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"Field {key!r} not in TensorFrame")
            from tensorframe.series import TensorSeries
            return TensorSeries(
                data={key: (self._data[key], self._schema.fields[key])},
                indices={
                    d: self._indices[d]
                    for d in self._schema.fields[key].dims
                    if d in self._indices
                },
            )
        elif isinstance(key, list):
            subset = {}
            for k in key:
                if k not in self._data:
                    raise KeyError(f"Field {k!r} not in TensorFrame")
                subset[k] = (self._data[k], self._schema.fields[k])
            relevant_dims = set()
            for k in key:
                relevant_dims.update(self._schema.fields[k].dims)
            return TensorFrame(
                data=subset,
                indices={
                    d: self._indices[d]
                    for d in relevant_dims
                    if d in self._indices
                },
                attrs=self._attrs,
            )
        raise TypeError(f"Key must be str or list[str], got {type(key)}")

    def get_array(self, field_name: str) -> jnp.ndarray:
        """Get the raw JAX array for a field."""
        if field_name not in self._data:
            raise KeyError(f"Field {field_name!r} not in TensorFrame")
        return self._data[field_name]

    # --- Indexing / Selection ---

    def isel(self, **dim_slices: int | slice | list[int] | np.ndarray) -> TensorFrame:
        """Select by integer position along named dimensions."""
        new_data = OrderedDict()
        new_indices: dict[str, Index] = {}

        # Track which dims are removed by integer indexing
        removed_dims: set[str] = set()
        for dim_name, sel in dim_slices.items():
            if isinstance(sel, (int, np.integer)):
                removed_dims.add(dim_name)

        for name, fspec in self._schema.fields.items():
            arr = self._data[name]
            indexer: list[Any] = [slice(None)] * arr.ndim
            for dim_name, sel in dim_slices.items():
                if dim_name not in fspec.dims:
                    continue
                axis = fspec.dims.index(dim_name)
                indexer[axis] = sel
            sliced = arr[tuple(indexer)]
            # Remove dims that were indexed with a scalar int
            new_dims = tuple(d for d in fspec.dims if d not in removed_dims)
            new_spec = FieldSpec(
                name=name,
                dtype=fspec.dtype,
                dims=new_dims,
                shape=tuple(sliced.shape),
                nullable=fspec.nullable,
                metadata=fspec.metadata,
            )
            new_data[name] = (sliced, new_spec)

        # Update indices
        for dim_name, idx in self._indices.items():
            if dim_name in dim_slices:
                sel = dim_slices[dim_name]
                if isinstance(sel, int):
                    continue  # dimension is removed
                elif isinstance(sel, slice):
                    new_labels = idx.labels[sel]
                elif isinstance(sel, (list, np.ndarray)):
                    new_labels = idx.labels[np.asarray(sel)]
                else:
                    new_labels = idx.labels[sel]
                new_indices[dim_name] = Index(labels=new_labels, name=idx.name)
            else:
                new_indices[dim_name] = idx

        return TensorFrame(data=new_data, indices=new_indices, attrs=self._attrs)

    def sel(self, **dim_labels: Any) -> TensorFrame:
        """Select by label along named dimensions.

        Supports single labels, slices (inclusive stop), and lists of labels.
        """
        int_slices = {}
        for dim_name, label in dim_labels.items():
            if dim_name not in self._indices:
                raise DimensionError(
                    f"Dimension {dim_name!r} not found in TensorFrame indices"
                )
            idx = self._indices[dim_name]
            if isinstance(label, slice):
                start, stop = idx.slice_locs(label.start, label.stop)
                int_slices[dim_name] = slice(start, stop)
            elif isinstance(label, (list, np.ndarray)):
                positions = idx.get_locs(label)
                int_slices[dim_name] = positions
            else:
                pos = idx.get_loc(label)
                int_slices[dim_name] = pos
        return self.isel(**int_slices)

    def where(self, mask: jnp.ndarray, dim: str | None = None) -> TensorFrame:
        """Boolean indexing: keep elements where mask is True.

        Parameters
        ----------
        mask : boolean jax.Array
            1D boolean mask along the specified dimension.
        dim : str, optional
            Dimension to filter. If None, uses the first shared dimension.
        """
        if dim is None:
            dim = self._dim_order[0]

        np_mask = np.asarray(mask)
        positions = np.where(np_mask)[0]
        return self.isel(**{dim: positions})

    # --- Transformation operations ---

    def with_column(
        self,
        name: str,
        data: jnp.ndarray | tuple[jnp.ndarray, FieldSpec],
    ) -> TensorFrame:
        """Return a new TensorFrame with an added or replaced field."""
        new_data = OrderedDict()
        for k in self._schema.field_names:
            new_data[k] = (self._data[k], self._schema.fields[k])
        new_data[name] = data if isinstance(data, tuple) else data
        return TensorFrame(
            data=new_data,
            indices=dict(self._indices),
            attrs=self._attrs,
        )

    def drop_fields(self, names: list[str]) -> TensorFrame:
        """Return a new TensorFrame without the specified fields."""
        new_data = OrderedDict()
        for k in self._schema.field_names:
            if k not in names:
                new_data[k] = (self._data[k], self._schema.fields[k])
        if not new_data:
            raise SchemaValidationError("Cannot drop all fields from TensorFrame")
        relevant_dims = set()
        for k, (_, spec) in new_data.items():
            relevant_dims.update(spec.dims)
        return TensorFrame(
            data=new_data,
            indices={d: self._indices[d] for d in relevant_dims if d in self._indices},
            attrs=self._attrs,
        )

    def rename_dims(self, mapping: dict[str, str]) -> TensorFrame:
        """Return a new TensorFrame with renamed dimensions."""
        new_data = OrderedDict()
        for name, fspec in self._schema.fields.items():
            new_dims = tuple(mapping.get(d, d) for d in fspec.dims)
            new_spec = FieldSpec(
                name=name,
                dtype=fspec.dtype,
                dims=new_dims,
                shape=fspec.shape,
                nullable=fspec.nullable,
                metadata=fspec.metadata,
            )
            new_data[name] = (self._data[name], new_spec)

        new_indices = {}
        for dim_name, idx in self._indices.items():
            new_name = mapping.get(dim_name, dim_name)
            new_indices[new_name] = idx.rename(new_name)

        return TensorFrame(data=new_data, indices=new_indices, attrs=self._attrs)

    def apply(
        self,
        fn: Callable[[jnp.ndarray], jnp.ndarray],
        fields: list[str] | None = None,
    ) -> TensorFrame:
        """Apply a JAX function to selected fields, returning a new TensorFrame."""
        target_fields = fields if fields is not None else self.field_names
        new_data = OrderedDict()
        for name in self._schema.field_names:
            if name in target_fields:
                result = fn(self._data[name])
                new_data[name] = result
            else:
                new_data[name] = (self._data[name], self._schema.fields[name])
        return TensorFrame(
            data=new_data,
            indices=dict(self._indices),
            attrs=self._attrs,
        )

    # --- Phase 3: Operations (delegated) ---

    def groupby(self, by: str) -> Any:
        """Group by a field, returning a GroupBy object for aggregation."""
        from tensorframe.ops import GroupBy
        return GroupBy(self, by)

    def map(self, fn: Callable, dim: str | None = None) -> jnp.ndarray:
        """Apply fn to each slice along a dimension, stacking results."""
        from tensorframe.ops import map_over_dim
        if dim is None:
            dim = self._dim_order[0]
        return map_over_dim(self, fn, dim)

    # --- Phase 4: ML Pipeline (delegated) ---

    def dropna(self, dim: str | None = None, fields: list[str] | None = None) -> TensorFrame:
        """Drop elements with NaN values."""
        from tensorframe.ml import dropna
        return dropna(self, dim=dim, fields=fields)

    def fillna(self, fill_values: dict[str, Any]) -> TensorFrame:
        """Replace NaN values in specified fields."""
        from tensorframe.ml import fillna
        return fillna(self, fill_values)

    def normalize(
        self,
        fields: list[str],
        method: str = "zscore",
        dim: str | tuple[str, ...] | None = None,
        return_params: bool = False,
    ) -> TensorFrame | tuple[TensorFrame, dict]:
        """Normalize selected fields."""
        from tensorframe.ml import normalize
        return normalize(self, fields, method=method, dim=dim, return_params=return_params)

    def encode_categorical(self, field_name: str, categories: list | None = None) -> TensorFrame:
        """Encode a field as integer category codes."""
        from tensorframe.ml import encode_categorical
        return encode_categorical(self, field_name, categories=categories)

    def one_hot(self, field_name: str, num_classes: int | None = None) -> TensorFrame:
        """One-hot encode a categorical integer field."""
        from tensorframe.ml import one_hot
        return one_hot(self, field_name, num_classes=num_classes)

    def split(
        self, dim: str | None = None, ratios: list[float] | None = None,
        shuffle: bool = True, seed: int = 42,
    ) -> tuple[TensorFrame, ...]:
        """Split into train/val/test subsets."""
        from tensorframe.ml import split
        return split(self, dim=dim, ratios=ratios, shuffle=shuffle, seed=seed)

    def to_jax_arrays(self, features: list[str], target: str | None = None):
        """Extract feature and target arrays for ML."""
        from tensorframe.ml import to_jax_arrays
        return to_jax_arrays(self, features, target=target)

    def iter_batches(
        self, batch_size: int, dim: str | None = None,
        shuffle: bool = False, seed: int = 0, drop_last: bool = False,
    ):
        """Iterate over the frame in batches."""
        from tensorframe.ml import iter_batches
        return iter_batches(self, batch_size, dim=dim, shuffle=shuffle, seed=seed, drop_last=drop_last)

    # --- Phase 2: Storage (delegated) ---

    def save(self, path: str, chunk_config: dict | None = None) -> None:
        """Save to Zarr v3 format on disk."""
        from tensorframe.storage import save
        save(self, path, chunk_config=chunk_config)

    # --- Conversion ---

    def to_dict(self) -> dict[str, jnp.ndarray]:
        """Return a dict of field_name -> jax.Array."""
        return dict(self._data)

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Return a dict of field_name -> numpy.ndarray."""
        return {k: np.asarray(v) for k, v in self._data.items()}

    # --- Representation ---

    def __repr__(self) -> str:
        lines = [f"TensorFrame(fields={self.num_fields}, dims={self.dims})"]
        for name, fspec in self._schema.fields.items():
            arr = self._data[name]
            lines.append(f"  {name}: {fspec.dtype} {arr.shape}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Length along the first shared dimension."""
        if not self._data:
            return 0
        first_arr = next(iter(self._data.values()))
        return first_arr.shape[0]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorFrame):
            return NotImplemented
        if self._schema != other._schema:
            return False
        for name in self._schema.field_names:
            if not jnp.array_equal(self._data[name], other._data[name]):
                return False
        return True


# --- JAX Pytree registration ---

def _tree_flatten(frame: TensorFrame) -> tuple[list[jnp.ndarray], Any]:
    """Flatten TensorFrame into JAX leaves (arrays) and aux data (metadata)."""
    leaves = [frame._data[name] for name in frame._schema.field_names]
    aux = (frame._schema, frame._indices, frame._dim_order, frame._attrs)
    return leaves, aux


def _tree_unflatten(aux: Any, leaves: list[jnp.ndarray]) -> TensorFrame:
    """Reconstruct TensorFrame from leaves and aux data."""
    schema, indices, dim_order, attrs = aux
    frame = object.__new__(TensorFrame)
    data = OrderedDict()
    for i, name in enumerate(schema.field_names):
        data[name] = leaves[i]
    object.__setattr__(frame, "_schema", schema)
    object.__setattr__(frame, "_data", data)
    object.__setattr__(frame, "_indices", indices)
    object.__setattr__(frame, "_dim_order", dim_order)
    object.__setattr__(frame, "_attrs", attrs)
    return frame


jax.tree_util.register_pytree_node(TensorFrame, _tree_flatten, _tree_unflatten)
