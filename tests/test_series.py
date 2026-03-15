"""Tests for TensorSeries."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from tensorframe.series import TensorSeries
from tensorframe.schema import FieldSpec
from tensorframe.ndtype import float32


class TestTensorSeries:
    def test_creation(self):
        series = TensorSeries(
            data={"vals": (jnp.array([1.0, 2.0, 3.0]), FieldSpec(
                name="vals", dtype=float32, dims=("x",), shape=(3,),
            ))},
        )
        assert series.name == "vals"
        assert series.num_fields == 1

    def test_multiple_fields_raises(self):
        with pytest.raises(ValueError, match="exactly 1 field"):
            TensorSeries(data={
                "a": jnp.array([1.0]),
                "b": jnp.array([2.0]),
            })

    def test_values(self):
        series = TensorSeries(
            data={"v": (jnp.array([10.0, 20.0, 30.0]), FieldSpec(
                name="v", dtype=float32, dims=("i",), shape=(3,),
            ))},
        )
        np.testing.assert_allclose(series.values, [10.0, 20.0, 30.0])

    def test_to_jax(self):
        series = TensorSeries(
            data={"v": (jnp.array([1.0, 2.0]), FieldSpec(
                name="v", dtype=float32, dims=("i",), shape=(2,),
            ))},
        )
        arr = series.to_jax()
        assert isinstance(arr, jax.Array)

    def test_dtype(self):
        series = TensorSeries(
            data={"v": (jnp.array([1.0]), FieldSpec(
                name="v", dtype=float32, dims=("i",), shape=(1,),
            ))},
        )
        assert series.dtype == float32

    def test_pytree_roundtrip(self):
        series = TensorSeries(
            data={"v": (jnp.array([1.0, 2.0, 3.0]), FieldSpec(
                name="v", dtype=float32, dims=("i",), shape=(3,),
            ))},
        )
        leaves, aux = jax.tree_util.tree_flatten(series)
        restored = jax.tree_util.tree_unflatten(aux, leaves)
        assert isinstance(restored, TensorSeries)
        np.testing.assert_allclose(restored.values, series.values)

    def test_repr(self):
        series = TensorSeries(
            data={"v": (jnp.array([1.0, 2.0]), FieldSpec(
                name="v", dtype=float32, dims=("i",), shape=(2,),
            ))},
        )
        r = repr(series)
        assert "TensorSeries" in r
        assert "v" in r

    def test_jit_on_series(self):
        series = TensorSeries(
            data={"v": (jnp.array([1.0, 4.0, 9.0]), FieldSpec(
                name="v", dtype=float32, dims=("i",), shape=(3,),
            ))},
        )

        @jax.jit
        def sqrt_series(s):
            return jax.tree.map(jnp.sqrt, s)

        result = sqrt_series(series)
        assert isinstance(result, TensorSeries)
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])
