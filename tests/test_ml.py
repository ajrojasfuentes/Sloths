"""Tests for Phase 4: ML Pipeline (normalize, split, encode, iter_batches)."""

import pytest
import numpy as np
import jax.numpy as jnp

from tensorframe.frame import TensorFrame
from tensorframe.schema import FieldSpec
from tensorframe.ndtype import float32, int32
from tensorframe.ml import (
    dropna,
    fillna,
    normalize,
    encode_categorical,
    one_hot,
    split,
    to_jax_arrays,
    iter_batches,
)
from tensorframe.errors import ShapeError


def make_ml_frame(n=100):
    rng = np.random.default_rng(42)
    return TensorFrame(data={
        "f1": (jnp.array(rng.standard_normal(n), dtype=jnp.float32), FieldSpec(
            name="f1", dtype=float32, dims=("batch",), shape=(n,))),
        "f2": (jnp.array(rng.standard_normal(n), dtype=jnp.float32), FieldSpec(
            name="f2", dtype=float32, dims=("batch",), shape=(n,))),
        "f3": (jnp.array(rng.standard_normal(n), dtype=jnp.float32), FieldSpec(
            name="f3", dtype=float32, dims=("batch",), shape=(n,))),
        "label": (jnp.array(rng.integers(0, 3, n), dtype=jnp.int32), FieldSpec(
            name="label", dtype=int32, dims=("batch",), shape=(n,))),
    })


def make_frame_with_nans():
    return TensorFrame(data={
        "x": jnp.array([1.0, float("nan"), 3.0, float("nan"), 5.0]),
        "y": jnp.array([10.0, 20.0, float("nan"), 40.0, 50.0]),
    })


class TestDropna:
    def test_dropna_removes_nans(self):
        frame = make_frame_with_nans()
        result = dropna(frame)
        # Only rows 0 and 4 have no NaN in either x or y
        assert len(result) == 2

    def test_dropna_specific_fields(self):
        frame = make_frame_with_nans()
        result = dropna(frame, fields=["x"])
        # x has NaN at positions 1, 3 → 3 rows remain
        assert len(result) == 3

    def test_dropna_no_nans(self):
        frame = TensorFrame(data={
            "a": jnp.array([1.0, 2.0, 3.0]),
        })
        result = dropna(frame)
        assert len(result) == 3

    def test_dropna_via_method(self):
        frame = make_frame_with_nans()
        result = frame.dropna()
        assert len(result) == 2


class TestFillna:
    def test_fillna_basic(self):
        frame = make_frame_with_nans()
        result = fillna(frame, {"x": 0.0, "y": -1.0})
        x = np.asarray(result.get_array("x"))
        y = np.asarray(result.get_array("y"))
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert x[1] == 0.0
        assert y[2] == -1.0

    def test_fillna_partial(self):
        frame = make_frame_with_nans()
        result = fillna(frame, {"x": 99.0})
        x = np.asarray(result.get_array("x"))
        y = np.asarray(result.get_array("y"))
        assert x[1] == 99.0
        assert np.isnan(y[2])  # y not filled

    def test_fillna_via_method(self):
        frame = make_frame_with_nans()
        result = frame.fillna({"x": 0.0})
        assert not np.any(np.isnan(np.asarray(result.get_array("x"))))


class TestNormalize:
    def test_zscore(self):
        frame = make_ml_frame()
        result = normalize(frame, fields=["f1", "f2"], method="zscore")
        f1 = np.asarray(result.get_array("f1"))
        np.testing.assert_allclose(np.mean(f1), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(f1), 1.0, atol=1e-2)

    def test_minmax(self):
        frame = make_ml_frame()
        result = normalize(frame, fields=["f1"], method="minmax")
        f1 = np.asarray(result.get_array("f1"))
        assert f1.min() >= -1e-6
        assert f1.max() <= 1.0 + 1e-6

    def test_normalize_return_params(self):
        frame = make_ml_frame()
        result, params = normalize(
            frame, fields=["f1"], method="zscore", return_params=True,
        )
        assert "f1" in params
        assert "mean" in params["f1"]
        assert "std" in params["f1"]

    def test_normalize_preserves_untargeted_fields(self):
        frame = make_ml_frame()
        result = normalize(frame, fields=["f1"])
        np.testing.assert_array_equal(
            result.get_array("label"),
            frame.get_array("label"),
        )

    def test_normalize_invalid_method(self):
        frame = make_ml_frame()
        with pytest.raises(ValueError, match="method must be"):
            normalize(frame, fields=["f1"], method="invalid")

    def test_normalize_via_method(self):
        frame = make_ml_frame()
        result = frame.normalize(fields=["f1", "f2"])
        f1 = np.asarray(result.get_array("f1"))
        np.testing.assert_allclose(np.mean(f1), 0.0, atol=1e-5)

    def test_normalize_constant_field(self):
        frame = TensorFrame(data={
            "const": jnp.ones(10),
        })
        result = normalize(frame, fields=["const"], method="zscore")
        c = np.asarray(result.get_array("const"))
        assert not np.any(np.isnan(c))  # division by zero handled


class TestEncodeCategorical:
    def test_encode_basic(self):
        frame = TensorFrame(data={
            "color": jnp.array([0, 1, 2, 0, 1]),  # pretend int categories
        })
        result = encode_categorical(frame, "color", categories=[0, 1, 2])
        codes = np.asarray(result.get_array("color"))
        np.testing.assert_array_equal(codes, [0, 1, 2, 0, 1])

    def test_encode_auto_categories(self):
        frame = TensorFrame(data={
            "x": jnp.array([3, 1, 2, 1, 3]),
        })
        result = encode_categorical(frame, "x")
        codes = np.asarray(result.get_array("x"))
        # auto categories sorted: [1, 2, 3] → codes: [2, 0, 1, 0, 2]
        assert codes[0] == 2  # 3 is category index 2
        assert codes[1] == 0  # 1 is category index 0

    def test_encode_via_method(self):
        frame = TensorFrame(data={
            "c": jnp.array([10, 20, 10, 30]),
        })
        result = frame.encode_categorical("c")
        assert result.num_fields == 1


class TestOneHot:
    def test_one_hot_basic(self):
        frame = TensorFrame(data={
            "label": jnp.array([0, 1, 2, 1, 0]),
        })
        result = one_hot(frame, "label", num_classes=3)
        encoded = np.asarray(result.get_array("label"))
        assert encoded.shape == (5, 3)
        np.testing.assert_array_equal(encoded[0], [1, 0, 0])
        np.testing.assert_array_equal(encoded[2], [0, 0, 1])

    def test_one_hot_auto_classes(self):
        frame = TensorFrame(data={
            "label": jnp.array([0, 1, 2]),
        })
        result = one_hot(frame, "label")
        assert result.get_array("label").shape == (3, 3)

    def test_one_hot_via_method(self):
        frame = TensorFrame(data={
            "cat": jnp.array([0, 1, 0]),
        })
        result = frame.one_hot("cat", num_classes=2)
        assert result.get_array("cat").shape == (3, 2)


class TestSplit:
    def test_default_split(self):
        frame = make_ml_frame(100)
        train, val, test = split(frame)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_custom_ratios(self):
        frame = make_ml_frame(100)
        a, b = split(frame, ratios=[0.8, 0.2])
        assert len(a) == 80
        assert len(b) == 20

    def test_split_no_shuffle(self):
        frame = make_ml_frame(100)
        train1, _, _ = split(frame, shuffle=False)
        train2, _, _ = split(frame, shuffle=False)
        np.testing.assert_array_equal(
            train1.get_array("f1"),
            train2.get_array("f1"),
        )

    def test_split_deterministic(self):
        frame = make_ml_frame(100)
        train1, _, _ = split(frame, seed=123)
        train2, _, _ = split(frame, seed=123)
        np.testing.assert_array_equal(
            train1.get_array("f1"),
            train2.get_array("f1"),
        )

    def test_split_different_seeds(self):
        frame = make_ml_frame(100)
        train1, _, _ = split(frame, seed=1)
        train2, _, _ = split(frame, seed=2)
        # Very unlikely to be identical
        assert not np.array_equal(
            np.asarray(train1.get_array("f1")),
            np.asarray(train2.get_array("f1")),
        )

    def test_split_invalid_ratios(self):
        frame = make_ml_frame()
        with pytest.raises(ValueError, match="sum to 1.0"):
            split(frame, ratios=[0.5, 0.3])

    def test_split_via_method(self):
        frame = make_ml_frame(100)
        train, val, test = frame.split(ratios=[0.6, 0.2, 0.2])
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_preserves_all_rows(self):
        frame = make_ml_frame(50)
        parts = split(frame, ratios=[0.6, 0.4], shuffle=False)
        total = sum(len(p) for p in parts)
        assert total == 50


class TestToJaxArrays:
    def test_basic(self):
        frame = make_ml_frame(50)
        X, y = to_jax_arrays(frame, features=["f1", "f2", "f3"], target="label")
        assert X.shape == (50, 3)
        assert y.shape == (50,)

    def test_no_target(self):
        frame = make_ml_frame(50)
        (X,) = to_jax_arrays(frame, features=["f1", "f2"])
        assert X.shape == (50, 2)

    def test_single_feature(self):
        frame = make_ml_frame(50)
        X, y = to_jax_arrays(frame, features=["f1"], target="label")
        assert X.shape == (50, 1)

    def test_via_method(self):
        frame = make_ml_frame(50)
        X, y = frame.to_jax_arrays(features=["f1", "f2"], target="label")
        assert X.shape == (50, 2)


class TestIterBatches:
    def test_basic_iteration(self):
        frame = make_ml_frame(100)
        batches = list(iter_batches(frame, batch_size=32))
        assert len(batches) == 4  # 32+32+32+4
        assert len(batches[0]) == 32
        assert len(batches[-1]) == 4

    def test_drop_last(self):
        frame = make_ml_frame(100)
        batches = list(iter_batches(frame, batch_size=32, drop_last=True))
        assert len(batches) == 3  # drops the 4-element batch

    def test_exact_division(self):
        frame = make_ml_frame(64)
        batches = list(iter_batches(frame, batch_size=32))
        assert len(batches) == 2
        assert all(len(b) == 32 for b in batches)

    def test_shuffle(self):
        frame = make_ml_frame(100)
        batches_a = list(iter_batches(frame, batch_size=50, shuffle=True, seed=1))
        batches_b = list(iter_batches(frame, batch_size=50, shuffle=True, seed=2))
        # Different seeds → different order
        assert not np.array_equal(
            np.asarray(batches_a[0].get_array("f1")),
            np.asarray(batches_b[0].get_array("f1")),
        )

    def test_deterministic_shuffle(self):
        frame = make_ml_frame(100)
        batches_a = list(iter_batches(frame, batch_size=50, shuffle=True, seed=42))
        batches_b = list(iter_batches(frame, batch_size=50, shuffle=True, seed=42))
        np.testing.assert_array_equal(
            batches_a[0].get_array("f1"),
            batches_b[0].get_array("f1"),
        )

    def test_via_method(self):
        frame = make_ml_frame(100)
        batches = list(frame.iter_batches(batch_size=25))
        assert len(batches) == 4

    def test_batch_is_tensorframe(self):
        frame = make_ml_frame(50)
        for batch in iter_batches(frame, batch_size=10):
            assert isinstance(batch, TensorFrame)
            assert batch.num_fields == 4

    def test_batch_preserves_fields(self):
        frame = make_ml_frame(50)
        batch = next(iter(iter_batches(frame, batch_size=10)))
        assert set(batch.field_names) == set(frame.field_names)

    def test_empty_frame(self):
        frame = TensorFrame(data={
            "x": (jnp.array([], dtype=jnp.float32).reshape(0,), FieldSpec(
                name="x", dtype=float32, dims=("batch",), shape=(0,))),
        })
        batches = list(iter_batches(frame, batch_size=10))
        assert len(batches) == 0


class TestMLPipelineIntegration:
    """End-to-end ML pipeline tests combining multiple operations."""

    def test_full_pipeline(self):
        frame = make_ml_frame(200)

        # Normalize features
        frame = frame.normalize(fields=["f1", "f2", "f3"])

        # Split
        train, val, test = frame.split(ratios=[0.7, 0.15, 0.15])
        assert len(train) + len(val) + len(test) == 200

        # Extract arrays
        X_train, y_train = train.to_jax_arrays(
            features=["f1", "f2", "f3"], target="label"
        )
        assert X_train.shape == (140, 3)
        assert y_train.shape == (140,)

        # Iterate batches
        batches = list(train.iter_batches(batch_size=32))
        assert all(b.num_fields == 4 for b in batches)

    def test_pipeline_with_dropna_and_fill(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100).astype(np.float32)
        data[::7] = float("nan")  # inject some NaNs

        frame = TensorFrame(data={
            "val": jnp.array(data),
            "target": jnp.array(rng.integers(0, 2, 100), dtype=jnp.int32),
        })

        # Fill NaNs, normalize, split
        filled = frame.fillna({"val": 0.0})
        normed = filled.normalize(fields=["val"])
        train, test = normed.split(ratios=[0.8, 0.2])

        X, y = train.to_jax_arrays(features=["val"], target="target")
        assert X.shape[0] == 80
        assert not np.any(np.isnan(np.asarray(X)))
