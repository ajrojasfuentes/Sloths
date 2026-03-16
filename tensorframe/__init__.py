"""TensorFrame: Multidimensional labeled data structures on JAX."""

from tensorframe.ndtype import (
    NDType,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bfloat16,
    complex64,
    complex128,
    datetime64,
    timedelta64,
    string,
    fixed_string,
    tensor,
    list_,
    fixed_list,
    struct,
    nullable,
    categorical,
)
from tensorframe.schema import FieldSpec, DimSpec, NDSchema
from tensorframe.index import Index, RangeIndex
from tensorframe.frame import TensorFrame
from tensorframe.series import TensorSeries
from tensorframe.errors import (
    TensorFrameError,
    SchemaError,
    SchemaValidationError,
    SchemaMismatchError,
    StorageError,
    MaterializationError,
    PersistenceError,
    ComputeError,
    ShapeError,
    DtypeError,
    IndexLabelError,
    DimensionError,
)
from tensorframe.construction import field, tensor_field

# Phase 2: Storage
from tensorframe.storage import save, open, LazyTensorFrame

# Phase 3: Operations
from tensorframe.ops import concat, merge, GroupBy, KernelRegistry, register_kernel, get_kernel

# Phase 4: ML Pipeline
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

__version__ = "0.1.0"

__all__ = [
    # Types
    "NDType",
    "bool_", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bfloat16",
    "complex64", "complex128",
    "datetime64", "timedelta64",
    "string", "fixed_string",
    "tensor", "list_", "fixed_list", "struct",
    "nullable", "categorical",
    # Schema
    "FieldSpec", "DimSpec", "NDSchema",
    # Index
    "Index", "RangeIndex",
    # Core
    "TensorFrame", "TensorSeries",
    "field", "tensor_field",
    # Storage
    "save", "open", "LazyTensorFrame",
    # Operations
    "concat", "merge", "GroupBy",
    "KernelRegistry", "register_kernel", "get_kernel",
    # ML
    "dropna", "fillna", "normalize",
    "encode_categorical", "one_hot",
    "split", "to_jax_arrays", "iter_batches",
    # Errors
    "TensorFrameError", "SchemaError", "SchemaValidationError",
    "SchemaMismatchError", "StorageError", "MaterializationError",
    "PersistenceError", "ComputeError",
    "ShapeError", "DtypeError", "IndexLabelError", "DimensionError",
]
