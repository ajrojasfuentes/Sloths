"""Tests for the NDType system."""

import json
import pytest
import jax.numpy as jnp

from tensorframe.ndtype import (
    NDType,
    ScalarType,
    bool_,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64, bfloat16,
    complex64, complex128,
    datetime64, timedelta64,
    string, fixed_string,
    tensor, list_, fixed_list,
    struct, nullable, categorical,
    TemporalType, StringType, FixedStringType,
    TensorType, ListType, FixedListType,
    StructType, NullableType, CategoricalType,
)


class TestScalarTypes:
    """Test scalar type creation and properties."""

    @pytest.mark.parametrize("ndtype,expected_jax", [
        (bool_, jnp.bool_),
        (int8, jnp.int8),
        (int16, jnp.int16),
        (int32, jnp.int32),
        (int64, jnp.int64),
        (uint8, jnp.uint8),
        (uint16, jnp.uint16),
        (uint32, jnp.uint32),
        (uint64, jnp.uint64),
        (float16, jnp.float16),
        (float32, jnp.float32),
        (float64, jnp.float64),
        (bfloat16, jnp.bfloat16),
        (complex64, jnp.complex64),
        (complex128, jnp.complex128),
    ])
    def test_scalar_jax_dtype(self, ndtype, expected_jax):
        assert ndtype.jax_dtype == expected_jax

    @pytest.mark.parametrize("ndtype", [
        bool_, int8, int16, int32, int64,
        uint8, uint16, uint32, uint64,
        float16, float32, float64, bfloat16,
        complex64, complex128,
    ])
    def test_scalar_is_frozen(self, ndtype):
        assert isinstance(ndtype, ScalarType)
        with pytest.raises(AttributeError):
            ndtype.name = "changed"

    @pytest.mark.parametrize("ndtype", [
        bool_, int32, float64, complex128,
    ])
    def test_scalar_roundtrip_json(self, ndtype):
        serialized = ndtype.to_json()
        restored = NDType.from_json(serialized)
        assert restored == ndtype
        assert restored.name == ndtype.name

    def test_scalar_str(self):
        assert str(float32) == "float32"
        assert str(int64) == "int64"


class TestTemporalTypes:
    """Test datetime64 and timedelta64."""

    def test_datetime64_default(self):
        dt = datetime64()
        assert dt.name == "datetime64"
        assert dt.unit == "s"
        assert dt.jax_dtype == jnp.int64

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_datetime64_units(self, unit):
        dt = datetime64(unit)
        assert dt.unit == unit

    def test_datetime64_invalid_unit(self):
        with pytest.raises(ValueError, match="unit must be one of"):
            datetime64("h")

    def test_timedelta64_roundtrip(self):
        td = timedelta64("ms")
        serialized = td.to_json()
        restored = NDType.from_json(serialized)
        assert restored.name == "timedelta64"
        assert restored.unit == "ms"

    def test_temporal_str(self):
        assert str(datetime64("ns")) == "datetime64[ns]"
        assert str(timedelta64("ms")) == "timedelta64[ms]"


class TestStringTypes:
    def test_string_type(self):
        assert string.name == "string"
        assert string.jax_dtype is None

    def test_string_roundtrip(self):
        restored = NDType.from_json(string.to_json())
        assert restored == string

    def test_fixed_string(self):
        fs = fixed_string(32)
        assert fs.max_length == 32
        assert str(fs) == "fixed_string[32]"

    def test_fixed_string_roundtrip(self):
        fs = fixed_string(64)
        restored = NDType.from_json(fs.to_json())
        assert isinstance(restored, FixedStringType)
        assert restored.max_length == 64


class TestCompositeTypes:
    def test_tensor_type(self):
        t = tensor(float32, (224, 224, 3))
        assert isinstance(t, TensorType)
        assert t.inner_dtype == float32
        assert t.shape == (224, 224, 3)
        assert t.jax_dtype == jnp.float32

    def test_tensor_roundtrip(self):
        t = tensor(int8, (28, 28))
        restored = NDType.from_json(t.to_json())
        assert isinstance(restored, TensorType)
        assert restored.shape == (28, 28)
        assert restored.inner_dtype == int8

    def test_list_type(self):
        lt = list_(int32)
        assert isinstance(lt, ListType)
        assert lt.inner_type == int32
        assert lt.jax_dtype is None

    def test_list_roundtrip(self):
        lt = list_(float64)
        restored = NDType.from_json(lt.to_json())
        assert isinstance(restored, ListType)
        assert restored.inner_type == float64

    def test_fixed_list_type(self):
        fl = fixed_list(float32, 10)
        assert fl.length == 10
        assert fl.inner_type == float32

    def test_fixed_list_roundtrip(self):
        fl = fixed_list(int16, 5)
        restored = NDType.from_json(fl.to_json())
        assert isinstance(restored, FixedListType)
        assert restored.length == 5

    def test_struct_type(self):
        s = struct({"x": float32, "y": float32, "label": int32})
        assert isinstance(s, StructType)
        assert len(s.fields) == 3

    def test_struct_roundtrip(self):
        s = struct({"pos": tensor(float32, (3,)), "id": int64})
        restored = NDType.from_json(s.to_json())
        assert isinstance(restored, StructType)
        assert "pos" in restored.fields
        assert isinstance(restored.fields["pos"], TensorType)

    def test_nullable_type(self):
        nt = nullable(float64)
        assert isinstance(nt, NullableType)
        assert nt.inner_type == float64
        assert nt.jax_dtype == jnp.float64

    def test_nullable_roundtrip(self):
        nt = nullable(int32)
        restored = NDType.from_json(nt.to_json())
        assert isinstance(restored, NullableType)
        assert restored.inner_type == int32

    def test_categorical_type(self):
        ct = categorical(["red", "green", "blue"], ordered=True)
        assert isinstance(ct, CategoricalType)
        assert ct.categories == ("red", "green", "blue")
        assert ct.ordered is True
        assert ct.jax_dtype == jnp.int32

    def test_categorical_roundtrip(self):
        ct = categorical(["a", "b", "c"])
        restored = NDType.from_json(ct.to_json())
        assert isinstance(restored, CategoricalType)
        assert restored.categories == ("a", "b", "c")
        assert restored.ordered is False


class TestNestedComposite:
    """Test deeply nested type compositions."""

    def test_nullable_list_of_tensors(self):
        t = nullable(list_(tensor(float32, (3, 3))))
        serialized = t.to_json()
        restored = NDType.from_json(serialized)
        assert isinstance(restored, NullableType)
        assert isinstance(restored.inner_type, ListType)
        assert isinstance(restored.inner_type.inner_type, TensorType)
        assert restored.inner_type.inner_type.shape == (3, 3)

    def test_struct_with_nested_types(self):
        s = struct({
            "coords": tensor(float64, (3,)),
            "tags": list_(int32),
            "meta": struct({"name": string, "count": int64}),
        })
        restored = NDType.from_json(s.to_json())
        assert isinstance(restored.fields["coords"], TensorType)
        assert isinstance(restored.fields["tags"], ListType)
        assert isinstance(restored.fields["meta"], StructType)


class TestEdgeCases:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown NDType kind"):
            NDType.from_dict({"kind": "nonexistent"})

    def test_empty_struct(self):
        s = struct({})
        assert len(s.fields) == 0
        restored = NDType.from_json(s.to_json())
        assert len(restored.fields) == 0

    def test_categorical_from_list(self):
        ct = categorical(["x", "y"])
        assert ct.categories == ("x", "y")

    def test_categorical_from_tuple(self):
        ct = categorical(("x", "y"))
        assert ct.categories == ("x", "y")
