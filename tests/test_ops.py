"""Tests for Phase 3: Operations (groupby, merge, concat, map, kernel registry)."""

import pytest
import numpy as np
import jax.numpy as jnp

from tensorframe.frame import TensorFrame
from tensorframe.schema import FieldSpec
from tensorframe.ndtype import float32, int32
from tensorframe.index import Index
from tensorframe.ops import (
    concat,
    merge,
    GroupBy,
    map_over_dim,
    KernelRegistry,
    register_kernel,
    get_kernel,
)
from tensorframe.errors import SchemaMismatchError, DimensionError, ShapeError


def make_frame_a():
    return TensorFrame(
        data={
            "id": jnp.array([1, 2, 3, 4, 5]),
            "precio": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "categoria": jnp.array([0, 1, 0, 1, 0]),
        },
        indices={"dim_0": Index(labels=np.array(["a", "b", "c", "d", "e"]))},
    )


def make_frame_b():
    return TensorFrame(
        data={
            "id": jnp.array([3, 4, 5, 6, 7]),
            "precio": jnp.array([300.0, 400.0, 500.0, 600.0, 700.0]),
            "categoria": jnp.array([1, 0, 1, 0, 1]),
        },
        indices={"dim_0": Index(labels=np.array(["f", "g", "h", "i", "j"]))},
    )


class TestConcat:
    def test_concat_basic(self):
        a = make_frame_a()
        b = make_frame_b()
        result = concat([a, b], dim="dim_0")
        assert len(result) == 10
        assert result.num_fields == 3

    def test_concat_preserves_data(self):
        a = make_frame_a()
        b = make_frame_b()
        result = concat([a, b], dim="dim_0")
        prices = np.asarray(result.get_array("precio"))
        np.testing.assert_allclose(
            prices,
            [10.0, 20.0, 30.0, 40.0, 50.0, 300.0, 400.0, 500.0, 600.0, 700.0],
        )

    def test_concat_single(self):
        a = make_frame_a()
        result = concat([a], dim="dim_0")
        assert len(result) == len(a)

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot concatenate empty"):
            concat([], dim="dim_0")

    def test_concat_mismatched_fields_raises(self):
        a = make_frame_a()
        b = TensorFrame(data={
            "id": jnp.array([1, 2]),
            "other": jnp.array([3.0, 4.0]),
        })
        with pytest.raises(SchemaMismatchError):
            concat([a, b], dim="dim_0")

    def test_concat_indices(self):
        a = make_frame_a()
        b = make_frame_b()
        result = concat([a, b], dim="dim_0")
        labels = result.indices["dim_0"].labels
        assert len(labels) == 10

    def test_concat_nd(self):
        imgs_a = TensorFrame(data={
            "img": (jnp.zeros((3, 4, 4)), FieldSpec(
                name="img", dtype=float32, dims=("batch", "h", "w"),
                shape=(3, 4, 4))),
        })
        imgs_b = TensorFrame(data={
            "img": (jnp.ones((2, 4, 4)), FieldSpec(
                name="img", dtype=float32, dims=("batch", "h", "w"),
                shape=(2, 4, 4))),
        })
        result = concat([imgs_a, imgs_b], dim="batch")
        assert result.get_array("img").shape == (5, 4, 4)


class TestMerge:
    def test_inner_join(self):
        left = TensorFrame(data={
            "id": jnp.array([1, 2, 3, 4]),
            "name": jnp.array([10.0, 20.0, 30.0, 40.0]),
        })
        right = TensorFrame(data={
            "id": jnp.array([2, 3, 5]),
            "score": jnp.array([0.8, 0.9, 0.7]),
        })
        result = merge(left, right, on="id", how="inner")
        assert len(result) == 2  # ids 2, 3
        assert "name" in result
        assert "score" in result

    def test_left_join(self):
        left = TensorFrame(data={
            "id": jnp.array([1, 2, 3]),
            "val": jnp.array([10.0, 20.0, 30.0]),
        })
        right = TensorFrame(data={
            "id": jnp.array([2, 3]),
            "extra": jnp.array([200.0, 300.0]),
        })
        result = merge(left, right, on="id", how="left")
        assert len(result) == 3  # all left keys

    def test_invalid_how_raises(self):
        a = make_frame_a()
        b = make_frame_b()
        with pytest.raises(ValueError, match="how must be"):
            merge(a, b, on="id", how="cross")


class TestGroupBy:
    def test_groupby_creation(self):
        frame = make_frame_a()
        gb = frame.groupby("categoria")
        assert isinstance(gb, GroupBy)
        assert gb.n_groups == 2

    def test_groupby_agg_mean(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({"precio": "mean"})
        assert result.num_fields == 2  # categoria + precio
        assert len(result) == 2

    def test_groupby_agg_sum(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({"precio": "sum"})
        prices = np.asarray(result.get_array("precio"))
        # cat 0: 10+30+50=90, cat 1: 20+40=60
        assert set(prices.tolist()) == {90.0, 60.0}

    def test_groupby_agg_multiple(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({
            "precio": "mean",
            "id": "count",
        })
        assert result.num_fields == 3  # categoria, precio, id

    def test_groupby_agg_count(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({"id": "count"})
        counts = np.asarray(result.get_array("id"))
        assert sorted(counts.tolist()) == [2, 3]

    def test_groupby_agg_custom_fn(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({
            "precio": lambda x: jnp.median(x),
        })
        assert len(result) == 2

    def test_groupby_missing_key_raises(self):
        frame = make_frame_a()
        with pytest.raises(KeyError):
            frame.groupby("nonexistent")

    def test_groupby_unknown_agg_raises(self):
        frame = make_frame_a()
        with pytest.raises(ValueError, match="Unknown aggregation"):
            frame.groupby("categoria").agg({"precio": "median_not_real"})

    def test_groupby_missing_field_raises(self):
        frame = make_frame_a()
        with pytest.raises(KeyError):
            frame.groupby("categoria").agg({"nonexistent": "mean"})

    def test_groupby_groups_dict(self):
        frame = make_frame_a()
        gb = frame.groupby("categoria")
        groups = gb.groups
        assert len(groups) == 2
        assert all(isinstance(v, np.ndarray) for v in groups.values())

    def test_groupby_repr(self):
        frame = make_frame_a()
        gb = frame.groupby("categoria")
        assert "GroupBy" in repr(gb)

    def test_groupby_agg_first_last(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({"precio": "first"})
        assert len(result) == 2

    def test_groupby_agg_min_max(self):
        frame = make_frame_a()
        result = frame.groupby("categoria").agg({"precio": "min"})
        assert len(result) == 2

    def test_groupby_apply(self):
        frame = make_frame_a()

        def double_prices(group):
            return group.with_column("precio", group.get_array("precio") * 2)

        result = frame.groupby("categoria").apply(double_prices)
        assert len(result) == 5  # all rows back


class TestMap:
    def test_map_basic(self):
        frame = TensorFrame(data={
            "x": jnp.array([1.0, 2.0, 3.0]),
            "y": jnp.array([4.0, 5.0, 6.0]),
        })
        result = frame.map(
            lambda row: row.get_array("x") + row.get_array("y"),
            dim="dim_0",
        )
        np.testing.assert_allclose(result, [5.0, 7.0, 9.0])

    def test_map_via_method(self):
        frame = TensorFrame(data={
            "x": jnp.array([1.0, 4.0, 9.0]),
        })
        result = frame.map(lambda row: jnp.sqrt(row.get_array("x")))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_map_missing_dim_raises(self):
        frame = TensorFrame(data={"x": jnp.array([1.0])})
        with pytest.raises(DimensionError):
            map_over_dim(frame, lambda r: r, dim="nonexistent")


class TestKernelRegistry:
    def test_builtin_kernels(self):
        k = get_kernel("sum")
        assert k.name == "sum"
        result = k(jnp.array([1.0, 2.0, 3.0]))
        assert float(result) == 6.0

    def test_register_custom(self):
        registry = KernelRegistry()
        registry.register("my_fn", lambda x: x ** 2, "Square elements")
        k = registry.get("my_fn")
        result = k(jnp.array([3.0]))
        np.testing.assert_allclose(result, [9.0])

    def test_missing_kernel_raises(self):
        registry = KernelRegistry()
        with pytest.raises(KeyError):
            registry.get("nope")

    def test_registry_contains(self):
        assert "mean" in get_kernel.__self__ if hasattr(get_kernel, '__self__') else True
        k = get_kernel("mean")
        assert k.name == "mean"

    def test_list_kernels(self):
        registry = KernelRegistry()
        registry.register("a", lambda x: x)
        registry.register("b", lambda x: x)
        assert set(registry.list_kernels()) == {"a", "b"}

    def test_registry_repr(self):
        registry = KernelRegistry()
        registry.register("a", lambda x: x)
        assert "1 kernels" in repr(registry)

    def test_kernel_repr(self):
        k = get_kernel("sum")
        assert "sum" in repr(k)
