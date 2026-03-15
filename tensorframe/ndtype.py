"""NDType system: type descriptors for TensorFrame fields.

Maps TensorFrame types to JAX dtypes and provides composite types
(tensor, list, struct, nullable, categorical) for rich data modeling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class NDType:
    """Base type descriptor for TensorFrame fields."""

    name: str
    jax_dtype: Any = None  # jnp.dtype or None for composite types

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.name}
        if self.jax_dtype is not None:
            d["jax_dtype"] = str(self.jax_dtype)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NDType:
        kind = d["kind"]
        factory = _REGISTRY.get(kind)
        if factory is not None:
            return factory(d)
        raise ValueError(f"Unknown NDType kind: {kind}")

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(s: str) -> NDType:
        return NDType.from_dict(json.loads(s))

    def __str__(self) -> str:
        return self.name


# --- Scalar types ---

@dataclass(frozen=True)
class ScalarType(NDType):
    """Scalar numeric or boolean type with a direct JAX dtype mapping."""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name}


def _make_scalar(name: str, jax_dt: Any) -> ScalarType:
    return ScalarType(name=name, jax_dtype=jax_dt)


bool_ = _make_scalar("bool_", jnp.bool_)
int8 = _make_scalar("int8", jnp.int8)
int16 = _make_scalar("int16", jnp.int16)
int32 = _make_scalar("int32", jnp.int32)
int64 = _make_scalar("int64", jnp.int64)
uint8 = _make_scalar("uint8", jnp.uint8)
uint16 = _make_scalar("uint16", jnp.uint16)
uint32 = _make_scalar("uint32", jnp.uint32)
uint64 = _make_scalar("uint64", jnp.uint64)
float16 = _make_scalar("float16", jnp.float16)
float32 = _make_scalar("float32", jnp.float32)
float64 = _make_scalar("float64", jnp.float64)
bfloat16 = _make_scalar("bfloat16", jnp.bfloat16)
complex64 = _make_scalar("complex64", jnp.complex64)
complex128 = _make_scalar("complex128", jnp.complex128)

_SCALAR_BY_NAME: dict[str, ScalarType] = {
    t.name: t
    for t in [
        bool_, int8, int16, int32, int64,
        uint8, uint16, uint32, uint64,
        float16, float32, float64, bfloat16,
        complex64, complex128,
    ]
}


# --- Temporal types ---

@dataclass(frozen=True)
class TemporalType(NDType):
    """Datetime or timedelta stored as int64 internally."""

    unit: str = "s"

    def __post_init__(self) -> None:
        valid_units = ("s", "ms", "us", "ns")
        if self.unit not in valid_units:
            raise ValueError(f"unit must be one of {valid_units}, got {self.unit!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "unit": self.unit}

    def __str__(self) -> str:
        return f"{self.name}[{self.unit}]"


def datetime64(unit: str = "s") -> TemporalType:
    return TemporalType(name="datetime64", jax_dtype=jnp.int64, unit=unit)


def timedelta64(unit: str = "s") -> TemporalType:
    return TemporalType(name="timedelta64", jax_dtype=jnp.int64, unit=unit)


# --- String types ---

@dataclass(frozen=True)
class StringType(NDType):
    """Variable-length UTF-8 string (offsets + bytes buffer)."""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "string"}


string = StringType(name="string", jax_dtype=None)


@dataclass(frozen=True)
class FixedStringType(NDType):
    """Fixed-length string of N bytes."""

    max_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fixed_string", "max_length": self.max_length}

    def __str__(self) -> str:
        return f"fixed_string[{self.max_length}]"


def fixed_string(n: int) -> FixedStringType:
    return FixedStringType(name="fixed_string", jax_dtype=None, max_length=n)


# --- Composite types ---

@dataclass(frozen=True)
class TensorType(NDType):
    """Field containing a sub-tensor of fixed shape per element."""

    inner_dtype: NDType = field(default_factory=lambda: float32)
    shape: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "tensor",
            "inner_dtype": self.inner_dtype.to_dict(),
            "shape": list(self.shape),
        }

    def __str__(self) -> str:
        return f"tensor[{self.inner_dtype}, {self.shape}]"


def tensor(inner_dtype: NDType, shape: tuple[int, ...]) -> TensorType:
    return TensorType(name="tensor", jax_dtype=inner_dtype.jax_dtype, inner_dtype=inner_dtype, shape=shape)


@dataclass(frozen=True)
class ListType(NDType):
    """Variable-length list of elements (offsets + flat values)."""

    inner_type: NDType = field(default_factory=lambda: float32)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "list_", "inner_type": self.inner_type.to_dict()}

    def __str__(self) -> str:
        return f"list_[{self.inner_type}]"


def list_(inner_type: NDType) -> ListType:
    return ListType(name="list_", jax_dtype=None, inner_type=inner_type)


@dataclass(frozen=True)
class FixedListType(NDType):
    """Fixed-length list of N elements."""

    inner_type: NDType = field(default_factory=lambda: float32)
    length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "fixed_list",
            "inner_type": self.inner_type.to_dict(),
            "length": self.length,
        }

    def __str__(self) -> str:
        return f"fixed_list[{self.inner_type}, {self.length}]"


def fixed_list(inner_type: NDType, length: int) -> FixedListType:
    return FixedListType(name="fixed_list", jax_dtype=inner_type.jax_dtype, inner_type=inner_type, length=length)


@dataclass(frozen=True)
class StructType(NDType):
    """Struct: named sub-fields, each stored as independent array."""

    fields: dict[str, NDType] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "struct",
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }

    def __str__(self) -> str:
        inner = ", ".join(f"{k}: {v}" for k, v in self.fields.items())
        return f"struct[{{{inner}}}]"


def struct(fields: dict[str, NDType]) -> StructType:
    return StructType(name="struct", jax_dtype=None, fields=fields)


@dataclass(frozen=True)
class NullableType(NDType):
    """Wraps any type to add null support via a validity bitmap."""

    inner_type: NDType = field(default_factory=lambda: float32)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "nullable", "inner_type": self.inner_type.to_dict()}

    def __str__(self) -> str:
        return f"nullable[{self.inner_type}]"


def nullable(inner_type: NDType) -> NullableType:
    return NullableType(name="nullable", jax_dtype=inner_type.jax_dtype, inner_type=inner_type)


@dataclass(frozen=True)
class CategoricalType(NDType):
    """Categorical: integer indices + category table."""

    categories: tuple[str, ...] = ()
    ordered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "categorical",
            "categories": list(self.categories),
            "ordered": self.ordered,
        }

    def __str__(self) -> str:
        return f"categorical[{list(self.categories)}, ordered={self.ordered}]"


def categorical(categories: list[str] | tuple[str, ...], ordered: bool = False) -> CategoricalType:
    cats = tuple(categories) if isinstance(categories, list) else categories
    return CategoricalType(name="categorical", jax_dtype=jnp.int32, categories=cats, ordered=ordered)


# --- Deserialization registry ---

def _scalar_from_dict(d: dict[str, Any]) -> ScalarType:
    return _SCALAR_BY_NAME[d["kind"]]


def _temporal_from_dict(d: dict[str, Any]) -> TemporalType:
    if d["kind"] == "datetime64":
        return datetime64(d.get("unit", "s"))
    return timedelta64(d.get("unit", "s"))


def _string_from_dict(_d: dict[str, Any]) -> StringType:
    return string


def _fixed_string_from_dict(d: dict[str, Any]) -> FixedStringType:
    return fixed_string(d["max_length"])


def _tensor_from_dict(d: dict[str, Any]) -> TensorType:
    return tensor(NDType.from_dict(d["inner_dtype"]), tuple(d["shape"]))


def _list_from_dict(d: dict[str, Any]) -> ListType:
    return list_(NDType.from_dict(d["inner_type"]))


def _fixed_list_from_dict(d: dict[str, Any]) -> FixedListType:
    return fixed_list(NDType.from_dict(d["inner_type"]), d["length"])


def _struct_from_dict(d: dict[str, Any]) -> StructType:
    return struct({k: NDType.from_dict(v) for k, v in d["fields"].items()})


def _nullable_from_dict(d: dict[str, Any]) -> NullableType:
    return nullable(NDType.from_dict(d["inner_type"]))


def _categorical_from_dict(d: dict[str, Any]) -> CategoricalType:
    return categorical(d["categories"], d.get("ordered", False))


_REGISTRY: dict[str, Any] = {}
for _name in _SCALAR_BY_NAME:
    _REGISTRY[_name] = _scalar_from_dict
_REGISTRY["datetime64"] = _temporal_from_dict
_REGISTRY["timedelta64"] = _temporal_from_dict
_REGISTRY["string"] = _string_from_dict
_REGISTRY["fixed_string"] = _fixed_string_from_dict
_REGISTRY["tensor"] = _tensor_from_dict
_REGISTRY["list_"] = _list_from_dict
_REGISTRY["fixed_list"] = _fixed_list_from_dict
_REGISTRY["struct"] = _struct_from_dict
_REGISTRY["nullable"] = _nullable_from_dict
_REGISTRY["categorical"] = _categorical_from_dict
