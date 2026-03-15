"""Tests for TensorFrame: core operations, pytree, indexing."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from tensorframe.frame import TensorFrame
from tensorframe.series import TensorSeries
from tensorframe.index import Index, RangeIndex
from tensorframe.schema import FieldSpec
from tensorframe.ndtype import float32, float64, int32
from tensorframe.construction import field, tensor_field
from tensorframe.errors import (
    ShapeError,
    DimensionError,
    SchemaValidationError,
    IndexLabelError,
)


# --- Fixtures ---

def make_simple_frame():
    """Create a simple 2-field TensorFrame for testing."""
    return TensorFrame({
        "precio": (jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]), FieldSpec(
            name="precio", dtype=float64,
            dims=("batch",), shape=(5,),
        )),
        "cantidad": (jnp.array([1, 2, 3, 4, 5]), FieldSpec(
            name="cantidad", dtype=int32,
            dims=("batch",), shape=(5,),
        )),
    }, indices={"batch": Index(labels=np.array(["a", "b", "c", "d", "e"]), name="batch")})


def make_nd_frame():
    """Create a multi-dimensional TensorFrame."""
    imgs = jnp.zeros((4, 8, 8, 3))
    labels = jnp.array([0, 1, 0, 1])
    return TensorFrame({
        "imagen": (imgs, FieldSpec(
            name="imagen", dtype=float32,
            dims=("batch", "h", "w", "c"), shape=(4, 8, 8, 3),
        )),
        "etiqueta": (labels, FieldSpec(
            name="etiqueta", dtype=int32,
            dims=("batch",), shape=(4,),
        )),
    })


class TestConstruction:
    def test_basic_construction(self):
        frame = make_simple_frame()
        assert frame.num_fields == 2
        assert "precio" in frame
        assert "cantidad" in frame

    def test_auto_infer_from_raw_arrays(self):
        frame = TensorFrame({
            "x": jnp.array([1.0, 2.0, 3.0]),
            "y": jnp.array([4.0, 5.0, 6.0]),
        })
        assert frame.num_fields == 2
        assert frame.dims == ("dim_0",)

    def test_construction_with_field_helper(self):
        arr_p, spec_p = field(
            [10.0, 20.0, 30.0], dims=("batch",), dtype=float64, name="precio"
        )
        arr_c, spec_c = field(
            [1, 2, 3], dims=("batch",), dtype=int32, name="cantidad"
        )
        frame = TensorFrame({
            "precio": (arr_p, spec_p),
            "cantidad": (arr_c, spec_c),
        })
        assert frame.num_fields == 2

    def test_construction_with_tensor_field(self):
        arr, spec = tensor_field(
            jnp.zeros((10, 28, 28)),
            dims=("batch", "h", "w"),
            name="imagen",
        )
        frame = TensorFrame({"imagen": (arr, spec)})
        assert frame.num_fields == 1

    def test_dimension_conflict_raises(self):
        with pytest.raises(ShapeError, match="conflicting sizes"):
            TensorFrame({
                "a": (jnp.zeros((5,)), FieldSpec(
                    name="a", dtype=float32, dims=("x",), shape=(5,))),
                "b": (jnp.zeros((10,)), FieldSpec(
                    name="b", dtype=float32, dims=("x",), shape=(10,))),
            })

    def test_index_dim_mismatch_raises(self):
        with pytest.raises(DimensionError, match="not in schema"):
            TensorFrame(
                data={"x": jnp.array([1.0, 2.0])},
                indices={"nonexistent": Index(labels=np.array([0, 1]))},
            )

    def test_index_length_mismatch_raises(self):
        with pytest.raises(ShapeError, match="length"):
            TensorFrame(
                data={"x": (jnp.array([1.0, 2.0, 3.0]), FieldSpec(
                    name="x", dtype=float32, dims=("batch",), shape=(3,)))},
                indices={"batch": Index(labels=np.array([0, 1]))},
            )

    def test_auto_range_index(self):
        frame = TensorFrame({
            "x": (jnp.array([1.0, 2.0, 3.0]), FieldSpec(
                name="x", dtype=float32, dims=("batch",), shape=(3,))),
        })
        assert "batch" in frame.indices
        assert isinstance(frame.indices["batch"], RangeIndex)
        assert len(frame.indices["batch"]) == 3


class TestProperties:
    def test_field_names(self):
        frame = make_simple_frame()
        assert frame.field_names == ["precio", "cantidad"]

    def test_dims(self):
        frame = make_simple_frame()
        assert frame.dims == ("batch",)

    def test_shape(self):
        frame = make_simple_frame()
        assert frame.shape == {"batch": 5}

    def test_len(self):
        frame = make_simple_frame()
        assert len(frame) == 5

    def test_contains(self):
        frame = make_simple_frame()
        assert "precio" in frame
        assert "nonexistent" not in frame

    def test_attrs(self):
        frame = TensorFrame(
            data={"x": jnp.array([1.0])},
            attrs={"source": "test"},
        )
        assert frame.attrs["source"] == "test"

    def test_nd_frame_dims(self):
        frame = make_nd_frame()
        assert "batch" in frame.dims
        assert "h" in frame.dims


class TestImmutability:
    def test_setattr_raises(self):
        frame = make_simple_frame()
        with pytest.raises(AttributeError, match="immutable"):
            frame._data = {}

    def test_delattr_raises(self):
        frame = make_simple_frame()
        with pytest.raises(AttributeError, match="immutable"):
            del frame._data


class TestFieldAccess:
    def test_getitem_single_returns_series(self):
        frame = make_simple_frame()
        result = frame["precio"]
        assert isinstance(result, TensorSeries)
        assert result.name == "precio"

    def test_getitem_list_returns_frame(self):
        frame = make_simple_frame()
        result = frame[["precio", "cantidad"]]
        assert isinstance(result, TensorFrame)
        assert result.num_fields == 2

    def test_getitem_missing_raises(self):
        frame = make_simple_frame()
        with pytest.raises(KeyError):
            frame["nonexistent"]

    def test_getitem_list_missing_raises(self):
        frame = make_simple_frame()
        with pytest.raises(KeyError):
            frame[["precio", "nonexistent"]]

    def test_getitem_invalid_type_raises(self):
        frame = make_simple_frame()
        with pytest.raises(TypeError):
            frame[42]

    def test_get_array(self):
        frame = make_simple_frame()
        arr = frame.get_array("precio")
        assert isinstance(arr, jax.Array)
        assert arr.shape == (5,)


class TestIsel:
    def test_isel_int(self):
        frame = make_simple_frame()
        result = frame.isel(batch=2)
        # Single int removes the dimension for each field
        assert result.get_array("precio").shape == ()

    def test_isel_slice(self):
        frame = make_simple_frame()
        result = frame.isel(batch=slice(1, 3))
        assert result.get_array("precio").shape == (2,)
        assert len(result) == 2

    def test_isel_list(self):
        frame = make_simple_frame()
        result = frame.isel(batch=[0, 2, 4])
        assert result.get_array("precio").shape == (3,)

    def test_isel_preserves_other_fields(self):
        frame = make_simple_frame()
        result = frame.isel(batch=slice(0, 3))
        assert result.num_fields == 2
        assert "precio" in result
        assert "cantidad" in result

    def test_isel_nd(self):
        frame = make_nd_frame()
        result = frame.isel(batch=slice(0, 2), h=slice(0, 4))
        assert result.get_array("imagen").shape == (2, 4, 8, 3)
        assert result.get_array("etiqueta").shape == (2,)

    def test_isel_updates_index(self):
        frame = make_simple_frame()
        result = frame.isel(batch=slice(1, 4))
        idx = result.indices["batch"]
        np.testing.assert_array_equal(idx.labels, ["b", "c", "d"])


class TestSel:
    def test_sel_single_label(self):
        frame = make_simple_frame()
        result = frame.sel(batch="c")
        assert result.get_array("precio").shape == ()

    def test_sel_slice(self):
        frame = make_simple_frame()
        result = frame.sel(batch=slice("b", "d"))
        assert result.get_array("precio").shape == (3,)
        np.testing.assert_allclose(
            result.get_array("precio"),
            jnp.array([20.0, 30.0, 40.0]),
        )

    def test_sel_list_of_labels(self):
        frame = make_simple_frame()
        result = frame.sel(batch=["a", "c", "e"])
        assert result.get_array("precio").shape == (3,)
        np.testing.assert_allclose(
            result.get_array("precio"),
            jnp.array([10.0, 30.0, 50.0]),
        )

    def test_sel_missing_dim_raises(self):
        frame = make_simple_frame()
        with pytest.raises(DimensionError):
            frame.sel(nonexistent="x")

    def test_sel_missing_label_raises(self):
        frame = make_simple_frame()
        with pytest.raises(IndexLabelError):
            frame.sel(batch="zzz")


class TestWhere:
    def test_where_boolean(self):
        frame = make_simple_frame()
        mask = frame.get_array("precio") > 25.0
        result = frame.where(mask)
        assert len(result) == 3
        np.testing.assert_allclose(
            result.get_array("precio"),
            jnp.array([30.0, 40.0, 50.0]),
        )


class TestWithColumn:
    def test_add_new_field(self):
        frame = make_simple_frame()
        new_arr = frame.get_array("precio") * 1.16
        result = frame.with_column("precio_iva", new_arr)
        assert result.num_fields == 3
        assert "precio_iva" in result
        # Original unchanged
        assert frame.num_fields == 2

    def test_replace_existing_field(self):
        frame = make_simple_frame()
        new_arr = frame.get_array("precio") * 2
        result = frame.with_column("precio", new_arr)
        assert result.num_fields == 2
        np.testing.assert_allclose(
            result.get_array("precio"),
            jnp.array([20.0, 40.0, 60.0, 80.0, 100.0]),
        )

    def test_add_field_with_spec(self):
        frame = make_simple_frame()
        arr, spec = field([True, False, True, True, False], dims=("batch",), name="flag")
        result = frame.with_column("flag", (arr, spec))
        assert result.num_fields == 3


class TestDropFields:
    def test_drop_single(self):
        frame = make_simple_frame()
        result = frame.drop_fields(["precio"])
        assert result.num_fields == 1
        assert "precio" not in result

    def test_drop_all_raises(self):
        frame = make_simple_frame()
        with pytest.raises(SchemaValidationError, match="Cannot drop all"):
            frame.drop_fields(["precio", "cantidad"])

    def test_drop_preserves_other_fields(self):
        frame = make_simple_frame()
        result = frame.drop_fields(["precio"])
        assert "cantidad" in result
        np.testing.assert_array_equal(
            result.get_array("cantidad"),
            jnp.array([1, 2, 3, 4, 5]),
        )


class TestRenameDims:
    def test_rename(self):
        frame = make_simple_frame()
        result = frame.rename_dims({"batch": "sample"})
        assert "sample" in result.dims
        assert "batch" not in result.dims
        assert "sample" in result.indices
        assert result.indices["sample"].name == "sample"


class TestApply:
    def test_apply_all_fields(self):
        frame = TensorFrame({
            "a": jnp.array([1.0, 4.0, 9.0]),
            "b": jnp.array([16.0, 25.0, 36.0]),
        })
        result = frame.apply(jnp.sqrt)
        np.testing.assert_allclose(result.get_array("a"), jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result.get_array("b"), jnp.array([4.0, 5.0, 6.0]))

    def test_apply_selected_fields(self):
        frame = TensorFrame({
            "a": jnp.array([1.0, 4.0, 9.0]),
            "b": jnp.array([16.0, 25.0, 36.0]),
        })
        result = frame.apply(jnp.sqrt, fields=["a"])
        np.testing.assert_allclose(result.get_array("a"), jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result.get_array("b"), jnp.array([16.0, 25.0, 36.0]))


class TestConversion:
    def test_to_dict(self):
        frame = make_simple_frame()
        d = frame.to_dict()
        assert "precio" in d
        assert isinstance(d["precio"], jax.Array)

    def test_to_numpy(self):
        frame = make_simple_frame()
        d = frame.to_numpy()
        assert isinstance(d["precio"], np.ndarray)
        np.testing.assert_allclose(d["precio"], [10.0, 20.0, 30.0, 40.0, 50.0])


class TestPytree:
    """Test JAX pytree registration and compatibility."""

    def test_tree_flatten_unflatten_roundtrip(self):
        frame = make_simple_frame()
        leaves, aux = jax.tree_util.tree_flatten(frame)
        restored = jax.tree_util.tree_unflatten(aux, leaves)
        assert restored.num_fields == frame.num_fields
        assert restored.field_names == frame.field_names
        for name in frame.field_names:
            np.testing.assert_array_equal(
                restored.get_array(name), frame.get_array(name)
            )

    def test_jit_identity(self):
        frame = make_simple_frame()

        @jax.jit
        def identity(f):
            return f

        result = identity(frame)
        assert result.num_fields == frame.num_fields
        np.testing.assert_allclose(
            result.get_array("precio"), frame.get_array("precio")
        )

    def test_jit_computation(self):
        frame = make_simple_frame()

        @jax.jit
        def double_precio(f):
            leaves, aux = jax.tree_util.tree_flatten(f)
            # Double all leaves
            new_leaves = [leaf * 2 for leaf in leaves]
            return jax.tree_util.tree_unflatten(aux, new_leaves)

        result = double_precio(frame)
        np.testing.assert_allclose(
            result.get_array("precio"),
            jnp.array([20.0, 40.0, 60.0, 80.0, 100.0]),
        )

    def test_tree_map(self):
        frame = make_simple_frame()
        doubled = jax.tree.map(lambda x: x * 2, frame)
        np.testing.assert_allclose(
            doubled.get_array("precio"),
            jnp.array([20.0, 40.0, 60.0, 80.0, 100.0]),
        )

    def test_vmap_over_leaves(self):
        """Test that vmap works on functions accessing pytree leaves."""
        frame = make_simple_frame()
        leaves, _ = jax.tree_util.tree_flatten(frame)
        # vmap over the batch dimension of each leaf
        result = jax.vmap(lambda x: x * 3)(leaves[0])
        np.testing.assert_allclose(result, jnp.array([30.0, 60.0, 90.0, 120.0, 150.0]))

    def test_nd_frame_pytree_roundtrip(self):
        frame = make_nd_frame()
        leaves, aux = jax.tree_util.tree_flatten(frame)
        assert len(leaves) == 2  # imagen + etiqueta
        restored = jax.tree_util.tree_unflatten(aux, leaves)
        assert restored.get_array("imagen").shape == (4, 8, 8, 3)


class TestRepr:
    def test_repr_shows_fields(self):
        frame = make_simple_frame()
        r = repr(frame)
        assert "TensorFrame" in r
        assert "precio" in r
        assert "cantidad" in r

    def test_repr_empty(self):
        # Cannot create empty frame, but we can test repr of minimal frame
        frame = TensorFrame({"x": jnp.array([1.0])})
        assert "TensorFrame" in repr(frame)


class TestEquality:
    def test_equal_frames(self):
        f1 = make_simple_frame()
        f2 = make_simple_frame()
        assert f1 == f2

    def test_different_data(self):
        f1 = make_simple_frame()
        f2 = f1.with_column("precio", f1.get_array("precio") * 2)
        assert f1 != f2

    def test_not_equal_to_other_type(self):
        frame = make_simple_frame()
        assert frame != "not a frame"
