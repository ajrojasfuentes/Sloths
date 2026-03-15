"""Tests for NDSchema, FieldSpec, and DimSpec."""

import json
import pytest
from collections import OrderedDict

from tensorframe.ndtype import float32, float64, int32, int64, tensor, nullable
from tensorframe.schema import NDSchema, FieldSpec, DimSpec
from tensorframe.errors import SchemaValidationError


class TestDimSpec:
    def test_creation(self):
        d = DimSpec(name="batch", size=64)
        assert d.name == "batch"
        assert d.size == 64

    def test_dynamic_size(self):
        d = DimSpec(name="time", size=None)
        assert d.size is None

    def test_roundtrip(self):
        d = DimSpec(name="x", size=100)
        restored = DimSpec.from_dict(d.to_dict())
        assert restored == d

    def test_repr(self):
        d = DimSpec(name="batch", size=32)
        assert "batch" in repr(d)
        assert "32" in repr(d)

    def test_dynamic_repr(self):
        d = DimSpec(name="time", size=None)
        assert "?" in repr(d)


class TestFieldSpec:
    def test_creation(self):
        f = FieldSpec(
            name="precio",
            dtype=float64,
            dims=("batch",),
            shape=(100,),
        )
        assert f.name == "precio"
        assert f.dtype == float64
        assert f.dims == ("batch",)
        assert f.shape == (100,)

    def test_dims_shape_mismatch_raises(self):
        with pytest.raises(SchemaValidationError, match="dims length"):
            FieldSpec(
                name="bad",
                dtype=float32,
                dims=("batch", "time"),
                shape=(100,),
            )

    def test_nullable_field(self):
        f = FieldSpec(
            name="optional_val",
            dtype=nullable(float32),
            dims=("batch",),
            shape=(50,),
            nullable=True,
        )
        assert f.nullable is True

    def test_metadata(self):
        f = FieldSpec(
            name="x",
            dtype=int32,
            dims=("i",),
            shape=(10,),
            metadata={"source": "sensor_a"},
        )
        assert f.metadata["source"] == "sensor_a"

    def test_roundtrip(self):
        f = FieldSpec(
            name="imagen",
            dtype=tensor(float32, (224, 224, 3)),
            dims=("batch", "h", "w", "c"),
            shape=(64, 224, 224, 3),
            nullable=False,
            metadata={"format": "RGB"},
        )
        restored = FieldSpec.from_dict(f.to_dict())
        assert restored.name == f.name
        assert restored.dims == f.dims
        assert restored.shape == f.shape
        assert restored.metadata == f.metadata


class TestNDSchema:
    def _make_simple_schema(self):
        fields = OrderedDict([
            ("precio", FieldSpec(
                name="precio", dtype=float64,
                dims=("batch",), shape=(100,),
            )),
            ("cantidad", FieldSpec(
                name="cantidad", dtype=int32,
                dims=("batch",), shape=(100,),
            )),
        ])
        dims = OrderedDict([
            ("batch", DimSpec(name="batch", size=100)),
        ])
        return NDSchema(fields=fields, dims=dims)

    def test_creation(self):
        schema = self._make_simple_schema()
        assert schema.num_fields == 2
        assert schema.field_names == ["precio", "cantidad"]
        assert schema.dim_names == ["batch"]

    def test_get_field(self):
        schema = self._make_simple_schema()
        f = schema.get_field("precio")
        assert f.dtype == float64

    def test_get_field_missing_raises(self):
        schema = self._make_simple_schema()
        with pytest.raises(KeyError, match="nonexistent"):
            schema.get_field("nonexistent")

    def test_field_key_name_mismatch_raises(self):
        with pytest.raises(SchemaValidationError, match="does not match"):
            NDSchema(
                fields=OrderedDict([
                    ("wrong_key", FieldSpec(
                        name="real_name", dtype=float32,
                        dims=("x",), shape=(10,),
                    ))
                ]),
                dims=OrderedDict([("x", DimSpec(name="x", size=10))]),
            )

    def test_field_references_missing_dim_raises(self):
        with pytest.raises(SchemaValidationError, match="not declared"):
            NDSchema(
                fields=OrderedDict([
                    ("f", FieldSpec(
                        name="f", dtype=float32,
                        dims=("nonexistent_dim",), shape=(10,),
                    ))
                ]),
                dims=OrderedDict(),
            )

    def test_with_field(self):
        schema = self._make_simple_schema()
        new_field = FieldSpec(
            name="total", dtype=float64,
            dims=("batch",), shape=(100,),
        )
        new_schema = schema.with_field(new_field)
        assert new_schema.num_fields == 3
        assert "total" in new_schema.field_names
        # Original unchanged
        assert schema.num_fields == 2

    def test_with_field_adds_new_dim(self):
        schema = self._make_simple_schema()
        new_field = FieldSpec(
            name="series", dtype=float32,
            dims=("batch", "time"), shape=(100, 50),
        )
        new_schema = schema.with_field(new_field)
        assert "time" in new_schema.dim_names

    def test_drop_field(self):
        schema = self._make_simple_schema()
        new_schema = schema.drop_field("precio")
        assert new_schema.num_fields == 1
        assert "precio" not in new_schema.field_names

    def test_drop_nonexistent_raises(self):
        schema = self._make_simple_schema()
        with pytest.raises(KeyError):
            schema.drop_field("nope")

    def test_drop_field_removes_orphan_dims(self):
        fields = OrderedDict([
            ("a", FieldSpec(name="a", dtype=float32, dims=("x",), shape=(10,))),
            ("b", FieldSpec(name="b", dtype=float32, dims=("y",), shape=(5,))),
        ])
        dims = OrderedDict([
            ("x", DimSpec(name="x", size=10)),
            ("y", DimSpec(name="y", size=5)),
        ])
        schema = NDSchema(fields=fields, dims=dims)
        new_schema = schema.drop_field("b")
        assert "y" not in new_schema.dim_names
        assert "x" in new_schema.dim_names

    def test_rename_dims(self):
        schema = self._make_simple_schema()
        new_schema = schema.rename_dims({"batch": "sample"})
        assert "sample" in new_schema.dim_names
        assert "batch" not in new_schema.dim_names
        assert new_schema.fields["precio"].dims == ("sample",)

    def test_json_roundtrip(self):
        schema = self._make_simple_schema()
        json_str = schema.to_json()
        restored = NDSchema.from_json(json_str)
        assert restored == schema

    def test_json_roundtrip_complex_schema(self):
        fields = OrderedDict([
            ("imagen", FieldSpec(
                name="imagen",
                dtype=tensor(float32, (224, 224, 3)),
                dims=("batch", "h", "w", "c"),
                shape=(64, 224, 224, 3),
            )),
            ("etiqueta", FieldSpec(
                name="etiqueta", dtype=int32,
                dims=("batch",), shape=(64,),
            )),
        ])
        dims = OrderedDict([
            ("batch", DimSpec(name="batch", size=64)),
            ("h", DimSpec(name="h", size=224)),
            ("w", DimSpec(name="w", size=224)),
            ("c", DimSpec(name="c", size=3)),
        ])
        schema = NDSchema(fields=fields, dims=dims, metadata={"version": "1.0"})
        restored = NDSchema.from_json(schema.to_json())
        assert restored == schema
        assert restored.metadata["version"] == "1.0"

    def test_equality(self):
        s1 = self._make_simple_schema()
        s2 = self._make_simple_schema()
        assert s1 == s2

    def test_inequality(self):
        s1 = self._make_simple_schema()
        s2 = s1.drop_field("precio")
        assert s1 != s2

    def test_repr(self):
        schema = self._make_simple_schema()
        r = repr(schema)
        assert "precio" in r
        assert "cantidad" in r
