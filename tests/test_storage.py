"""Tests for Phase 2: Storage Layer (save/open, lazy loading)."""

import os
import tempfile
import shutil
import pytest
import numpy as np
import jax.numpy as jnp

from tensorframe.frame import TensorFrame
from tensorframe.schema import FieldSpec
from tensorframe.ndtype import float32, int32
from tensorframe.index import Index
from tensorframe.storage import save, open as tf_open, LazyTensorFrame
from tensorframe.errors import PersistenceError, StorageError


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_test_frame():
    return TensorFrame(
        data={
            "precio": (jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]), FieldSpec(
                name="precio", dtype=float32,
                dims=("batch",), shape=(5,),
            )),
            "cantidad": (jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32), FieldSpec(
                name="cantidad", dtype=int32,
                dims=("batch",), shape=(5,),
            )),
        },
        indices={"batch": Index(labels=np.array(["a", "b", "c", "d", "e"]), name="batch")},
    )


def make_nd_test_frame():
    return TensorFrame(
        data={
            "imagen": (jnp.ones((4, 8, 8, 3), dtype=jnp.float32), FieldSpec(
                name="imagen", dtype=float32,
                dims=("batch", "h", "w", "c"), shape=(4, 8, 8, 3),
            )),
            "etiqueta": (jnp.array([0, 1, 0, 1], dtype=jnp.int32), FieldSpec(
                name="etiqueta", dtype=int32,
                dims=("batch",), shape=(4,),
            )),
        },
    )


class TestSave:
    def test_save_basic(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "test.zarr")
        save(frame, path)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "zarr.json"))

    def test_save_nd(self, tmp_dir):
        frame = make_nd_test_frame()
        path = os.path.join(tmp_dir, "nd.zarr")
        save(frame, path)
        assert os.path.exists(path)

    def test_save_with_chunk_config(self, tmp_dir):
        frame = make_nd_test_frame()
        path = os.path.join(tmp_dir, "chunked.zarr")
        save(frame, path, chunk_config={
            "imagen": {"chunks": (2, 8, 8, 3)},
        })
        assert os.path.exists(path)

    def test_save_via_method(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "method.zarr")
        frame.save(path)
        assert os.path.exists(path)


class TestOpen:
    def test_open_roundtrip(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "roundtrip.zarr")
        save(frame, path)

        loaded = tf_open(path)
        assert loaded.num_fields == frame.num_fields
        assert loaded.field_names == frame.field_names
        np.testing.assert_allclose(
            loaded.get_array("precio"),
            frame.get_array("precio"),
        )
        np.testing.assert_array_equal(
            loaded.get_array("cantidad"),
            frame.get_array("cantidad"),
        )

    def test_open_nd_roundtrip(self, tmp_dir):
        frame = make_nd_test_frame()
        path = os.path.join(tmp_dir, "nd_rt.zarr")
        save(frame, path)

        loaded = tf_open(path)
        assert loaded.get_array("imagen").shape == (4, 8, 8, 3)
        np.testing.assert_allclose(
            loaded.get_array("imagen"),
            frame.get_array("imagen"),
        )

    def test_open_preserves_indices(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "idx.zarr")
        save(frame, path)

        loaded = tf_open(path)
        assert "batch" in loaded.indices
        np.testing.assert_array_equal(
            loaded.indices["batch"].labels,
            np.array(["a", "b", "c", "d", "e"]),
        )

    def test_open_nonexistent_raises(self):
        with pytest.raises(StorageError, match="does not exist"):
            tf_open("/tmp/nonexistent_path_xyz123")


class TestLazyLoading:
    def test_lazy_open(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        assert isinstance(lazy, LazyTensorFrame)
        assert lazy.num_fields == 2
        assert not lazy.is_cached("precio")

    def test_lazy_field_access_materializes(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_access.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        arr = lazy.get_array("precio")
        assert lazy.is_cached("precio")
        np.testing.assert_allclose(arr, [10.0, 20.0, 30.0, 40.0, 50.0])

    def test_lazy_getitem(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_get.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        arr = lazy["cantidad"]
        np.testing.assert_array_equal(arr, [1, 2, 3, 4, 5])

    def test_lazy_compute(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_compute.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        materialized = lazy.compute()
        assert isinstance(materialized, TensorFrame)
        assert materialized.num_fields == 2
        np.testing.assert_allclose(
            materialized.get_array("precio"),
            frame.get_array("precio"),
        )

    def test_lazy_evict(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_evict.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        lazy.get_array("precio")  # materialize
        assert lazy.is_cached("precio")
        lazy.evict("precio")
        assert not lazy.is_cached("precio")

    def test_lazy_evict_all(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_evict_all.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        lazy.get_array("precio")
        lazy.get_array("cantidad")
        lazy.evict_all()
        assert not lazy.is_cached("precio")
        assert not lazy.is_cached("cantidad")

    def test_lazy_properties(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_props.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        assert lazy.field_names == ["precio", "cantidad"]
        assert "batch" in lazy.dims
        assert lazy.shape == {"batch": 5}
        assert len(lazy) == 5
        assert "precio" in lazy

    def test_lazy_repr(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_repr.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        r = repr(lazy)
        assert "LazyTensorFrame" in r
        assert "0/2" in r

    def test_lazy_missing_field_raises(self, tmp_dir):
        frame = make_test_frame()
        path = os.path.join(tmp_dir, "lazy_missing.zarr")
        save(frame, path)

        lazy = tf_open(path, lazy=True)
        with pytest.raises(KeyError):
            lazy.get_array("nonexistent")


class TestSaveOpenConsistency:
    """Test that save → open preserves all data and metadata."""

    def test_multiple_save_open_cycles(self, tmp_dir):
        frame = make_test_frame()
        for i in range(3):
            path = os.path.join(tmp_dir, f"cycle_{i}.zarr")
            save(frame, path)
            frame = tf_open(path)

        np.testing.assert_allclose(
            frame.get_array("precio"),
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )

    def test_modified_frame_save_open(self, tmp_dir):
        frame = make_test_frame()
        modified = frame.with_column("doble", frame.get_array("precio") * 2)
        path = os.path.join(tmp_dir, "modified.zarr")
        save(modified, path)

        loaded = tf_open(path)
        assert "doble" in loaded
        np.testing.assert_allclose(
            loaded.get_array("doble"),
            [20.0, 40.0, 60.0, 80.0, 100.0],
        )
