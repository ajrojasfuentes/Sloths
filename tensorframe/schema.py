"""NDSchema: N-dimensional typed schema descriptor for TensorFrame."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
from typing import Any

from tensorframe.ndtype import NDType, float32
from tensorframe.errors import SchemaValidationError


@dataclass(frozen=True)
class DimSpec:
    """Specification for a single dimension."""

    name: str
    size: int | None = None  # None = dynamic

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DimSpec:
        return DimSpec(name=d["name"], size=d.get("size"))

    def __repr__(self) -> str:
        size_str = str(self.size) if self.size is not None else "?"
        return f"DimSpec({self.name!r}, size={size_str})"


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single field in the schema."""

    name: str
    dtype: NDType = dc_field(default_factory=lambda: float32)
    dims: tuple[str, ...] = ()
    shape: tuple[int | None, ...] = ()
    nullable: bool = False
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.dims) != len(self.shape):
            raise SchemaValidationError(
                f"Field {self.name!r}: dims length ({len(self.dims)}) must match "
                f"shape length ({len(self.shape)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.to_dict(),
            "dims": list(self.dims),
            "shape": [s if s is not None else None for s in self.shape],
            "nullable": self.nullable,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FieldSpec:
        return FieldSpec(
            name=d["name"],
            dtype=NDType.from_dict(d["dtype"]),
            dims=tuple(d["dims"]),
            shape=tuple(d["shape"]),
            nullable=d.get("nullable", False),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"FieldSpec({self.name!r}, dtype={self.dtype}, dims={self.dims}, shape={self.shape})"


@dataclass(frozen=True)
class NDSchema:
    """Complete typed schema for a TensorFrame.

    Describes all fields, their types, dimensions, and shapes.
    Analogous to Arrow Schema but for N-dimensional data.
    """

    fields: OrderedDict[str, FieldSpec] = dc_field(default_factory=OrderedDict)
    dims: OrderedDict[str, DimSpec] = dc_field(default_factory=OrderedDict)
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate internal consistency of the schema."""
        for fname, fspec in self.fields.items():
            if fspec.name != fname:
                raise SchemaValidationError(
                    f"Field key {fname!r} does not match FieldSpec.name {fspec.name!r}"
                )
            for dim_name in fspec.dims:
                if dim_name not in self.dims:
                    raise SchemaValidationError(
                        f"Field {fname!r} references dimension {dim_name!r} "
                        f"not declared in schema dims"
                    )

    @property
    def field_names(self) -> list[str]:
        return list(self.fields.keys())

    @property
    def dim_names(self) -> list[str]:
        return list(self.dims.keys())

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    def get_field(self, name: str) -> FieldSpec:
        if name not in self.fields:
            raise KeyError(f"Field {name!r} not in schema")
        return self.fields[name]

    def with_field(self, field_spec: FieldSpec) -> NDSchema:
        """Return new schema with an added or replaced field."""
        new_fields = OrderedDict(self.fields)
        new_fields[field_spec.name] = field_spec
        new_dims = OrderedDict(self.dims)
        for i, dim_name in enumerate(field_spec.dims):
            if dim_name not in new_dims:
                new_dims[dim_name] = DimSpec(name=dim_name, size=field_spec.shape[i])
        return NDSchema(fields=new_fields, dims=new_dims, metadata=self.metadata)

    def drop_field(self, name: str) -> NDSchema:
        """Return new schema without the named field."""
        if name not in self.fields:
            raise KeyError(f"Field {name!r} not in schema")
        new_fields = OrderedDict(
            (k, v) for k, v in self.fields.items() if k != name
        )
        used_dims = set()
        for f in new_fields.values():
            used_dims.update(f.dims)
        new_dims = OrderedDict(
            (k, v) for k, v in self.dims.items() if k in used_dims
        )
        return NDSchema(fields=new_fields, dims=new_dims, metadata=self.metadata)

    def rename_dims(self, mapping: dict[str, str]) -> NDSchema:
        """Return new schema with renamed dimensions."""
        new_dims = OrderedDict()
        for k, v in self.dims.items():
            new_name = mapping.get(k, k)
            new_dims[new_name] = DimSpec(name=new_name, size=v.size)
        new_fields = OrderedDict()
        for k, f in self.fields.items():
            new_field_dims = tuple(mapping.get(d, d) for d in f.dims)
            new_fields[k] = FieldSpec(
                name=f.name,
                dtype=f.dtype,
                dims=new_field_dims,
                shape=f.shape,
                nullable=f.nullable,
                metadata=f.metadata,
            )
        return NDSchema(fields=new_fields, dims=new_dims, metadata=self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "dims": {k: v.to_dict() for k, v in self.dims.items()},
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NDSchema:
        fields = OrderedDict(
            (k, FieldSpec.from_dict(v)) for k, v in d["fields"].items()
        )
        dims = OrderedDict(
            (k, DimSpec.from_dict(v)) for k, v in d["dims"].items()
        )
        return NDSchema(
            fields=fields,
            dims=dims,
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_json(s: str) -> NDSchema:
        return NDSchema.from_dict(json.loads(s))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NDSchema):
            return NotImplemented
        return (
            self.fields == other.fields
            and self.dims == other.dims
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        fields_str = ", ".join(
            f"{f.name}: {f.dtype}" for f in self.fields.values()
        )
        return f"NDSchema({{{fields_str}}})"
