"""Tests for the exception hierarchy."""

import pytest

from tensorframe.errors import (
    TensorFrameError,
    SchemaError,
    SchemaValidationError,
    SchemaMismatchError,
    SchemaEvolutionError,
    StorageError,
    MaterializationError,
    DeviceMemoryError,
    PersistenceError,
    ComputeError,
    ShapeError,
    DtypeError,
    JITTraceError,
    IndexLabelError,
    DimensionError,
)


class TestExceptionHierarchy:
    """Verify the exception hierarchy is correct."""

    def test_schema_errors_inherit_from_schema_error(self):
        assert issubclass(SchemaValidationError, SchemaError)
        assert issubclass(SchemaMismatchError, SchemaError)
        assert issubclass(SchemaEvolutionError, SchemaError)

    def test_schema_error_inherits_from_base(self):
        assert issubclass(SchemaError, TensorFrameError)

    def test_storage_errors_inherit(self):
        assert issubclass(MaterializationError, StorageError)
        assert issubclass(DeviceMemoryError, StorageError)
        assert issubclass(PersistenceError, StorageError)
        assert issubclass(StorageError, TensorFrameError)

    def test_compute_errors_inherit(self):
        assert issubclass(ShapeError, ComputeError)
        assert issubclass(DtypeError, ComputeError)
        assert issubclass(JITTraceError, ComputeError)
        assert issubclass(ComputeError, TensorFrameError)

    def test_index_errors_inherit(self):
        assert issubclass(IndexLabelError, TensorFrameError)
        assert issubclass(DimensionError, TensorFrameError)

    def test_all_catchable_by_base(self):
        errors = [
            SchemaValidationError("test"),
            SchemaMismatchError("test"),
            MaterializationError("test"),
            ShapeError("test"),
            IndexLabelError("test"),
            DimensionError("test"),
        ]
        for e in errors:
            with pytest.raises(TensorFrameError):
                raise e

    def test_error_messages(self):
        e = SchemaValidationError("dims mismatch")
        assert "dims mismatch" in str(e)
