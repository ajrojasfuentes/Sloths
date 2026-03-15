"""Tests for field construction helpers."""

import pytest
import jax.numpy as jnp
import numpy as np

from tensorframe.construction import field, tensor_field, _infer_dtype
from tensorframe.ndtype import float32, float64, int32, int64, TensorType


class TestInferDtype:
    def test_infer_float32(self):
        arr = jnp.array([1.0], dtype=jnp.float32)
        assert _infer_dtype(arr) == float32

    def test_infer_int32(self):
        arr = jnp.array([1], dtype=jnp.int32)
        inferred = _infer_dtype(arr)
        # JAX may default ints to int32; accept the actual dtype
        assert inferred.name == "int32"


class TestField:
    def test_from_list(self):
        arr, spec = field([1.0, 2.0, 3.0], dims=("x",))
        assert arr.shape == (3,)
        assert spec.dims == ("x",)

    def test_from_numpy(self):
        arr, spec = field(np.array([1, 2, 3]), dims=("batch",), name="vals")
        assert spec.name == "vals"

    def test_auto_dims(self):
        arr, spec = field(jnp.zeros((3, 4)))
        assert spec.dims == ("dim_0", "dim_1")

    def test_explicit_dtype(self):
        arr, spec = field([1, 2, 3], dims=("x",), dtype=float32)
        assert arr.dtype == jnp.float32
        assert spec.dtype == float32

    def test_dims_mismatch_raises(self):
        with pytest.raises(ValueError, match="dims length"):
            field(jnp.zeros((3, 4)), dims=("only_one",))


class TestTensorField:
    def test_basic(self):
        arr, spec = tensor_field(
            jnp.zeros((10, 28, 28)),
            dims=("batch", "h", "w"),
            name="image",
        )
        assert isinstance(spec.dtype, TensorType)
        assert spec.dtype.shape == (28, 28)
        assert arr.shape == (10, 28, 28)

    def test_1d_tensor_field(self):
        arr, spec = tensor_field(jnp.array([1.0, 2.0, 3.0]), dims=("x",))
        assert isinstance(spec.dtype, TensorType)
        assert spec.dtype.shape == ()

    def test_auto_dims(self):
        arr, spec = tensor_field(jnp.zeros((5, 3, 3)))
        assert spec.dims == ("dim_0", "dim_1", "dim_2")

    def test_dims_mismatch_raises(self):
        with pytest.raises(ValueError, match="dims length"):
            tensor_field(jnp.zeros((5, 3)), dims=("a", "b", "c"))
