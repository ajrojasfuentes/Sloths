"""Tests for Index and RangeIndex."""

import json
import pytest
import numpy as np

from tensorframe.index import Index, RangeIndex
from tensorframe.errors import IndexLabelError


class TestIndex:
    def test_creation_from_list(self):
        idx = Index(labels=["a", "b", "c"], name="letters")
        assert len(idx) == 3
        assert idx.name == "letters"

    def test_creation_from_numpy(self):
        idx = Index(labels=np.array([10, 20, 30]), name="ids")
        assert len(idx) == 3

    def test_1d_requirement(self):
        with pytest.raises(ValueError, match="1D"):
            Index(labels=np.array([[1, 2], [3, 4]]))

    def test_contains(self):
        idx = Index(labels=np.array([10, 20, 30]))
        assert 20 in idx
        assert 99 not in idx

    def test_get_loc(self):
        idx = Index(labels=np.array(["a", "b", "c"]))
        assert idx.get_loc("b") == 1

    def test_get_loc_missing_raises(self):
        idx = Index(labels=np.array([1, 2, 3]))
        with pytest.raises(IndexLabelError, match="not found"):
            idx.get_loc(99)

    def test_get_locs(self):
        idx = Index(labels=np.array([10, 20, 30, 40, 50]))
        positions = idx.get_locs([20, 40])
        np.testing.assert_array_equal(positions, [1, 3])

    def test_slice_locs(self):
        idx = Index(labels=np.array([10, 20, 30, 40, 50]))
        s, e = idx.slice_locs(20, 40)
        assert s == 1
        assert e == 4  # inclusive stop → position + 1

    def test_slice_locs_none_start(self):
        idx = Index(labels=np.array([10, 20, 30]))
        s, e = idx.slice_locs(None, 20)
        assert s == 0
        assert e == 2

    def test_slice_locs_none_stop(self):
        idx = Index(labels=np.array([10, 20, 30]))
        s, e = idx.slice_locs(10, None)
        assert s == 0
        assert e == 3

    def test_rename(self):
        idx = Index(labels=np.array([1, 2, 3]), name="old")
        new_idx = idx.rename("new")
        assert new_idx.name == "new"
        assert idx.name == "old"  # original unchanged

    def test_equality(self):
        idx1 = Index(labels=np.array([1, 2, 3]), name="x")
        idx2 = Index(labels=np.array([1, 2, 3]), name="x")
        assert idx1 == idx2

    def test_inequality_labels(self):
        idx1 = Index(labels=np.array([1, 2, 3]))
        idx2 = Index(labels=np.array([1, 2, 4]))
        assert idx1 != idx2

    def test_inequality_name(self):
        idx1 = Index(labels=np.array([1, 2, 3]), name="a")
        idx2 = Index(labels=np.array([1, 2, 3]), name="b")
        assert idx1 != idx2

    def test_json_roundtrip(self):
        idx = Index(labels=np.array([10, 20, 30]), name="ids")
        restored = Index.from_json(idx.to_json())
        assert restored == idx

    def test_json_roundtrip_string_labels(self):
        idx = Index(labels=np.array(["x", "y", "z"]), name="coords")
        restored = Index.from_json(idx.to_json())
        np.testing.assert_array_equal(restored.labels, idx.labels)

    def test_repr_short(self):
        idx = Index(labels=np.array([1, 2, 3]), name="x")
        r = repr(idx)
        assert "Index" in r
        assert "x" in r

    def test_repr_long(self):
        idx = Index(labels=np.arange(100))
        r = repr(idx)
        assert "..." in r

    def test_hash(self):
        idx1 = Index(labels=np.array([1, 2, 3]), name="x")
        idx2 = Index(labels=np.array([1, 2, 3]), name="x")
        assert hash(idx1) == hash(idx2)
        # Can be used in sets/dicts
        s = {idx1, idx2}
        assert len(s) == 1


class TestRangeIndex:
    def test_creation(self):
        idx = RangeIndex(10)
        assert len(idx) == 10
        np.testing.assert_array_equal(idx.labels, np.arange(10))

    def test_creation_with_start_step(self):
        idx = RangeIndex(start=5, stop=20, step=5)
        np.testing.assert_array_equal(idx.labels, [5, 10, 15])

    def test_name(self):
        idx = RangeIndex(5, name="batch")
        assert idx.name == "batch"

    def test_get_loc(self):
        idx = RangeIndex(start=0, stop=100, step=10)
        assert idx.get_loc(30) == 3

    def test_json_roundtrip(self):
        idx = RangeIndex(start=0, stop=50, step=2, name="even")
        data = idx.to_dict()
        restored = RangeIndex.from_dict(data)
        assert restored.start == 0
        assert restored.stop == 50
        assert restored.step == 2
        assert restored.name == "even"

    def test_from_index_from_dict_dispatches(self):
        """Index.from_dict should return RangeIndex when kind is range_index."""
        idx = RangeIndex(10, name="test")
        restored = Index.from_dict(idx.to_dict())
        assert isinstance(restored, RangeIndex)

    def test_repr(self):
        idx = RangeIndex(start=0, stop=10, step=1, name="i")
        r = repr(idx)
        assert "RangeIndex" in r
        assert "10" in r
