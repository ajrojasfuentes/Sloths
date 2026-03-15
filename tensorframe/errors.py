"""Exception hierarchy for TensorFrame."""


class TensorFrameError(Exception):
    """Base exception for all TensorFrame errors."""


# --- Schema errors ---

class SchemaError(TensorFrameError):
    """Base for schema-related errors."""


class SchemaValidationError(SchemaError):
    """Raised when a schema is structurally invalid."""


class SchemaMismatchError(SchemaError):
    """Raised when schemas are incompatible for an operation (merge, concat)."""


class SchemaEvolutionError(SchemaError):
    """Raised when schema migration fails."""


# --- Storage errors ---

class StorageError(TensorFrameError):
    """Base for storage-related errors."""


class MaterializationError(StorageError):
    """Raised when promoting data from Cold/Warm to Hot fails."""


class DeviceMemoryError(StorageError):
    """Raised when device memory is insufficient for materialization."""


class PersistenceError(StorageError):
    """Raised when writing to cold storage fails."""


# --- Compute errors ---

class ComputeError(TensorFrameError):
    """Base for compute-related errors."""


class ShapeError(ComputeError):
    """Raised when array shapes are incompatible for an operation."""


class DtypeError(ComputeError):
    """Raised when dtypes are incompatible for an operation."""


class JITTraceError(ComputeError):
    """Raised when JIT tracing fails on a TensorFrame operation."""


# --- Index errors ---

class IndexLabelError(TensorFrameError):
    """Raised when a label is not found in an Index."""


class DimensionError(TensorFrameError):
    """Raised when a referenced dimension does not exist."""
