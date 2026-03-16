# TensorFrame — Documentación Formal del Proyecto

**Proyecto:** Sloths / TensorFrame
**Versión:** 0.1.0
**Fecha:** Marzo 2026
**Estado:** Fases 0–4 implementadas (Fundamentos, Core, Storage, Operaciones, ML Pipeline)

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Diseño y Decisiones Técnicas](#3-diseño-y-decisiones-técnicas)
4. [Estado Actual del Desarrollo](#4-estado-actual-del-desarrollo)
5. [Manual de Usuario](#5-manual-de-usuario)
6. [Referencia de API](#6-referencia-de-api)
7. [Testing y Calidad](#7-testing-y-calidad)
8. [Guía de Contribución](#8-guía-de-contribución)

---

## 1. Visión General

### ¿Qué es TensorFrame?

TensorFrame es un framework de estructuras de datos etiquetadas y multidimensionales construido sobre JAX. Combina la ergonomía de Pandas con cómputo acelerado en GPU/TPU, un sistema de tipos rico inspirado en Apache Arrow, y un storage layer de tres niveles basado en Zarr v3 y TensorStore.

### Caso de Uso Principal

Herramienta de propósito general para:
- Análisis científico de datos multidimensionales
- Limpieza y normalización de datasets
- Preparación de datos para ML/DL
- Manipulación de datasets heterogéneos, multidimensionales y multimodales

### Stack Tecnológico

| Componente | Tecnología | Propósito |
|---|---|---|
| Cómputo | JAX (XLA) | GPU/TPU, JIT, autograd, vmap |
| Storage L1 | jax.Array | Datos en device memory |
| Storage L2 | TensorStore | I/O async, cache, vistas virtuales |
| Storage L3 | Zarr v3 | Persistencia chunked/comprimida |
| Lenguaje | Python ≥ 3.10 | API de usuario |

### Instalación

```bash
pip install -e ".[dev]"
```

Dependencias principales:
- `jax >= 0.4.20`
- `jaxlib >= 0.4.20`
- `zarr >= 3.0`
- `tensorstore >= 0.1.45`

---

## 2. Arquitectura del Sistema

### Arquitectura en Capas

```
┌─────────────────────────────────────────────────────────────────┐
│                        Capa de Usuario                          │
│  TensorFrame · TensorSeries · Pipeline API                     │
│  (indexing, groupby, merge, normalize, split, iter_batches)     │
├─────────────────────────────────────────────────────────────────┤
│                     Capa de Esquema y Tipos                     │
│  NDSchema · NDType · Index · RangeIndex · FieldSpec · DimSpec   │
│  (Definición de estructura, validación, serialización JSON)     │
├─────────────────────────────────────────────────────────────────┤
│                    Capa de Cómputo (JAX)                        │
│  Kernel Registry · Pytree Integration · JIT/vmap/grad          │
│  (Evaluación eager, transformaciones funcionales)               │
├─────────────────────────────────────────────────────────────────┤
│                 Capa de Storage (3 niveles)                      │
│  L1: JAX Array (Hot) → L2: TensorStore (Warm) → L3: Zarr (Cold)│
│  (Materialización on-demand, caching, persistencia)             │
├─────────────────────────────────────────────────────────────────┤
│                   Capa de Errores                                │
│  TensorFrameError → Schema/Storage/Compute/Index errors         │
│  (Jerarquía de excepciones con 13 tipos específicos)            │
└─────────────────────────────────────────────────────────────────┘
```

### Estructuras de Datos Principales

#### TensorFrame
Contenedor inmutable N-dimensional que agrupa campos (arrays JAX) que comparten al menos una dimensión de alineación. Registrado como pytree JAX para compatibilidad con `jit`, `grad`, `vmap`.

**Anatomía interna:**
```
TensorFrame
├── _schema: NDSchema         # Descriptor de estructura
├── _data: OrderedDict        # field_name → jax.Array
├── _indices: dict             # dim_name → Index
├── _dim_order: tuple          # Orden de dimensiones
└── _attrs: dict               # Metadatos de usuario
```

#### TensorSeries
Especialización de TensorFrame con exactamente un campo. Proporciona acceso directo via `.values` y `.to_jax()`.

#### NDSchema
Descriptor completo de la estructura tipada de un TensorFrame. Contiene `FieldSpec` (por campo) y `DimSpec` (por dimensión). Serializable a/desde JSON.

#### NDType
Sistema de tipos con 15 tipos escalares, 2 temporales, 2 de string, y 7 tipos compuestos (tensor, list, struct, nullable, categorical). Todos son inmutables (`frozen=True`) y serializables.

---

## 3. Diseño y Decisiones Técnicas

### Inmutabilidad Total
Todos los TensorFrames y sus campos son inmutables. Las operaciones retornan nuevos objetos. Esto es un requisito duro de JAX para `jit`, `grad` y `vmap`, y elimina bugs de aliasing.

### Un Campo = Un Array JAX
No hay consolidación por dtype (como el BlockManager de Pandas). XLA fusiona operaciones a nivel de compilación, haciendo redundante la consolidación en memoria.

### Pytree Registration
TensorFrame se registra como pytree JAX separando:
- **Hojas (datos dinámicos):** Arrays JAX de cada campo
- **Aux data (metadatos estáticos):** Schema, índices, dim_order, attrs

Esto permite usar TensorFrame directamente con `jax.jit`, `jax.grad`, `jax.vmap`.

### Storage Layer de 3 Niveles
- **L1 (Hot):** `jax.Array` en device memory, listo para cómputo
- **L2 (Warm):** TensorStore con cache y vistas virtuales
- **L3 (Cold):** Zarr v3 en disco/cloud, chunked y comprimido

Al abrir un dataset con `lazy=True`, todo empieza en Cold/Warm y se materializa on-demand.

### Eager para Cómputo, Lazy para I/O
Las operaciones aritméticas sobre arrays materializados se ejecutan inmediatamente (JAX estándar). La carga desde disco es lazy.

---

## 4. Estado Actual del Desarrollo

### Fases Completadas

| Fase | Estado | Módulos | Líneas de Código |
|---|---|---|---|
| **Fase 0: Fundamentos** | Completada | `ndtype.py`, `schema.py`, `index.py`, `errors.py` | ~580 |
| **Fase 1: Core Inmutable** | Completada | `frame.py`, `series.py`, `construction.py` | ~380 |
| **Fase 2: Storage** | Completada | `storage.py` | ~310 |
| **Fase 3: Operaciones** | Completada | `ops.py` | ~320 |
| **Fase 4: ML Pipeline** | Completada | `ml.py` | ~310 |

### Fases Pendientes

| Fase | Alcance |
|---|---|
| **Fase 5: Interop** | Arrow bridge, Pandas bridge, DLPack, Parquet/CSV I/O |
| **Fase 6: Avanzado** | LazyExpr graph, sharding automático, TensorBatch IPC, optimización |

### Métricas de Calidad

- **296 tests** pasando
- **93% code coverage** global
- **0 warnings** en el código propio (las 16 warnings son de Zarr/JAX)
- **13 tipos de excepción** con jerarquía formal
- **Serialización JSON** idempotente para tipos, esquemas e índices

### Estructura del Proyecto

```
Sloths/
├── tensorframe/
│   ├── __init__.py          # Exports públicos (64 símbolos)
│   ├── ndtype.py            # Sistema de tipos NDType
│   ├── schema.py            # NDSchema, FieldSpec, DimSpec
│   ├── index.py             # Index, RangeIndex
│   ├── errors.py            # Jerarquía de excepciones
│   ├── construction.py      # Helpers: field(), tensor_field()
│   ├── frame.py             # TensorFrame (core)
│   ├── series.py            # TensorSeries
│   ├── storage.py           # Save/Open Zarr v3, LazyTensorFrame
│   ├── ops.py               # GroupBy, concat, merge, map, KernelRegistry
│   └── ml.py                # normalize, split, encode, iter_batches
├── tests/
│   ├── test_ndtype.py       # 65 tests
│   ├── test_schema.py       # 26 tests
│   ├── test_index.py        # 26 tests
│   ├── test_errors.py       # 7 tests
│   ├── test_construction.py # 11 tests
│   ├── test_frame.py        # 57 tests
│   ├── test_series.py       # 8 tests
│   ├── test_storage.py      # 20 tests
│   ├── test_ops.py          # 30 tests
│   └── test_ml.py           # 46 tests
├── docs/
│   └── DOCUMENTATION.md     # Este documento
├── pyproject.toml            # Configuración del proyecto
├── TensorFrame_Diseno_Arquitectonico.md  # Documento de diseño
└── README.md
```

---

## 5. Manual de Usuario

### 5.1 Inicio Rápido

```python
import tensorframe as tf
import jax.numpy as jnp

# Crear un TensorFrame desde datos en memoria
frame = tf.TensorFrame({
    "precio": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    "cantidad": jnp.array([1, 2, 3, 4, 5]),
    "categoria": jnp.array([0, 1, 0, 1, 0]),
})

print(frame)
# TensorFrame(fields=3, dims=('dim_0',))
#   precio: float32 (5,)
#   cantidad: int32 (5,)
#   categoria: int32 (5,)
```

### 5.2 Construcción con Tipos Explícitos

```python
import numpy as np

# Usando helpers de construcción
arr_p, spec_p = tf.field([10.0, 20.0, 30.0], dims=("batch",), dtype=tf.float32, name="precio")
arr_c, spec_c = tf.field([1, 2, 3], dims=("batch",), dtype=tf.int32, name="cantidad")

frame = tf.TensorFrame({
    "precio": (arr_p, spec_p),
    "cantidad": (arr_c, spec_c),
})

# Con índices explícitos
frame = tf.TensorFrame(
    data={
        "precio": (jnp.array([10.0, 20.0, 30.0]), tf.FieldSpec(
            name="precio", dtype=tf.float32, dims=("batch",), shape=(3,))),
    },
    indices={"batch": tf.Index(labels=np.array(["a", "b", "c"]), name="batch")},
)
```

### 5.3 Datos Multidimensionales

```python
# TensorFrame con campos de diferentes dimensionalidades
frame = tf.TensorFrame({
    "imagen": (jnp.zeros((100, 28, 28)), tf.FieldSpec(
        name="imagen", dtype=tf.float32,
        dims=("batch", "h", "w"), shape=(100, 28, 28))),
    "etiqueta": (jnp.zeros(100, dtype=jnp.int32), tf.FieldSpec(
        name="etiqueta", dtype=tf.int32,
        dims=("batch",), shape=(100,))),
})
# imagen: (100, 28, 28), etiqueta: (100,)
# Comparten la dimensión "batch"
```

### 5.4 Indexing y Selección

```python
# Acceso a campos
series = frame["precio"]          # → TensorSeries
subset = frame[["precio", "cantidad"]]  # → TensorFrame

# Selección por posición (como .iloc de Pandas)
frame.isel(batch=0)              # primera fila
frame.isel(batch=slice(0, 10))   # primeras 10 filas
frame.isel(batch=[0, 5, 9])     # filas específicas

# Selección por etiqueta (como .loc de Pandas)
frame.sel(batch="sample_a")
frame.sel(batch=slice("a", "c"))  # inclusive
frame.sel(batch=["a", "c", "e"])

# Filtrado booleano
mask = frame.get_array("precio") > 25.0
frame.where(mask)
```

### 5.5 Transformaciones

```python
# Agregar/reemplazar campo (inmutable: retorna nuevo frame)
frame2 = frame.with_column("precio_iva", frame.get_array("precio") * 1.16)

# Eliminar campos
frame3 = frame.drop_fields(["temp_col"])

# Renombrar dimensiones
frame4 = frame.rename_dims({"batch": "sample"})

# Aplicar función JAX a campos
frame5 = frame.apply(jnp.log1p, fields=["precio"])
```

### 5.6 GroupBy y Agregaciones

```python
# Agrupar y agregar
result = frame.groupby("categoria").agg({
    "precio": "mean",
    "cantidad": "sum",
})

# Funciones de agregación disponibles:
# "mean", "sum", "min", "max", "std", "var", "count", "first", "last"

# Función personalizada
result = frame.groupby("categoria").agg({
    "precio": lambda x: jnp.median(x),
})

# Aplicar función por grupo
result = frame.groupby("categoria").apply(
    lambda group: group.with_column("precio", group.get_array("precio") * 2)
)
```

### 5.7 Concat y Merge

```python
# Concatenar frames
combined = tf.concat([frame_a, frame_b], dim="batch")

# Merge/Join por campo
result = tf.merge(left, right, on="id", how="inner")
# how: "inner", "left", "right", "outer"
```

### 5.8 Map sobre Dimensión

```python
# Aplicar función a cada elemento a lo largo de una dimensión
results = frame.map(
    lambda row: row.get_array("x") + row.get_array("y"),
    dim="batch",
)
# Retorna jax.Array con los resultados apilados
```

### 5.9 Pipeline ML

#### Manejo de NaN
```python
# Eliminar filas con NaN
clean = frame.dropna()
clean = frame.dropna(fields=["precio"])  # solo verificar ciertos campos

# Rellenar NaN
filled = frame.fillna({"precio": 0.0, "cantidad": -1})
```

#### Normalización
```python
# Z-score (mean=0, std=1)
normed = frame.normalize(fields=["f1", "f2"], method="zscore")

# Min-max (min=0, max=1)
normed = frame.normalize(fields=["f1"], method="minmax")

# Obtener parámetros para aplicar en test set
normed, params = frame.normalize(fields=["f1"], return_params=True)
# params = {"f1": {"mean": ..., "std": ...}}
```

#### Encoding Categórico
```python
# Enteros categóricos
encoded = frame.encode_categorical("color", categories=[0, 1, 2])

# One-hot encoding
one_hot_frame = frame.one_hot("label", num_classes=3)
```

#### Split Train/Val/Test
```python
train, val, test = frame.split(ratios=[0.7, 0.15, 0.15], shuffle=True, seed=42)
```

#### Extracción de Arrays para ML
```python
X, y = frame.to_jax_arrays(features=["f1", "f2", "f3"], target="label")
# X: (N, 3), y: (N,)
```

#### Batching
```python
for batch in frame.iter_batches(batch_size=64, shuffle=True):
    X, y = batch.to_jax_arrays(features=["f1", "f2"], target="label")
    # entrenar modelo...
```

### 5.10 Persistencia (Zarr v3)

```python
# Guardar a disco
frame.save("dataset.zarr")

# Con configuración de chunks
frame.save("dataset.zarr", chunk_config={
    "imagen": {"chunks": (64, 28, 28)},
})

# Abrir (eager: carga todo en memoria)
loaded = tf.open("dataset.zarr")

# Abrir lazy (carga on-demand)
lazy = tf.open("dataset.zarr", lazy=True)
print(lazy.is_cached("imagen"))  # False

arr = lazy.get_array("imagen")   # materializa este campo
print(lazy.is_cached("imagen"))  # True

# Materializar todo
full_frame = lazy.compute()
```

### 5.11 Compatibilidad con JAX

```python
import jax

# JIT compilation
@jax.jit
def normalizar(frame):
    return jax.tree.map(lambda x: x / x.max(), frame)

result = normalizar(frame)

# tree_map
doubled = jax.tree.map(lambda x: x * 2, frame)

# vmap sobre hojas
leaves, aux = jax.tree_util.tree_flatten(frame)
processed = jax.vmap(lambda x: x ** 2)(leaves[0])
```

### 5.12 Pipeline Completo de Ejemplo

```python
import tensorframe as tf
import jax.numpy as jnp
import numpy as np

# 1. Crear dataset
rng = np.random.default_rng(42)
n = 1000
frame = tf.TensorFrame({
    "feature_1": jnp.array(rng.standard_normal(n), dtype=jnp.float32),
    "feature_2": jnp.array(rng.standard_normal(n), dtype=jnp.float32),
    "feature_3": jnp.array(rng.standard_normal(n), dtype=jnp.float32),
    "label": jnp.array(rng.integers(0, 3, n), dtype=jnp.int32),
})

# 2. Normalizar features
frame = frame.normalize(fields=["feature_1", "feature_2", "feature_3"])

# 3. Split train/test
train, test = frame.split(ratios=[0.8, 0.2], seed=42)

# 4. Guardar a disco
train.save("train.zarr")
test.save("test.zarr")

# 5. Cargar y entrenar
train_loaded = tf.open("train.zarr")
for epoch in range(10):
    for batch in train_loaded.iter_batches(batch_size=64, shuffle=True, seed=epoch):
        X, y = batch.to_jax_arrays(
            features=["feature_1", "feature_2", "feature_3"],
            target="label",
        )
        # loss = train_step(model, X, y)

# 6. Evaluar
test_loaded = tf.open("test.zarr")
X_test, y_test = test_loaded.to_jax_arrays(
    features=["feature_1", "feature_2", "feature_3"],
    target="label",
)
```

---

## 6. Referencia de API

### Tipos (tensorframe.ndtype)

**Escalares:** `bool_`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`, `bfloat16`, `complex64`, `complex128`

**Temporales:** `datetime64(unit)`, `timedelta64(unit)` — unidades: `"s"`, `"ms"`, `"us"`, `"ns"`

**Strings:** `string`, `fixed_string(n)`

**Compuestos:**
- `tensor(inner_dtype, shape)` — sub-tensor por elemento
- `list_(inner_type)` — lista de longitud variable
- `fixed_list(inner_type, n)` — lista de longitud fija
- `struct(fields_dict)` — campos nombrados
- `nullable(inner_type)` — tipo con soporte de nulls
- `categorical(categories, ordered)` — encoding categórico

Todos los tipos soportan `.to_json()` / `.from_json()` para serialización.

### Schema (tensorframe.schema)

| Clase | Descripción |
|---|---|
| `DimSpec(name, size)` | Especificación de dimensión |
| `FieldSpec(name, dtype, dims, shape, nullable, metadata)` | Especificación de campo |
| `NDSchema(fields, dims, metadata)` | Esquema completo con validación |

Métodos de NDSchema: `with_field()`, `drop_field()`, `rename_dims()`, `to_json()`, `from_json()`

### Index (tensorframe.index)

| Clase | Descripción |
|---|---|
| `Index(labels, name)` | Índice basado en etiquetas (NumPy) |
| `RangeIndex(stop, start, step, name)` | Índice eficiente para rangos enteros |

Métodos: `get_loc()`, `get_locs()`, `slice_locs()`, `rename()`, `to_json()`, `from_json()`

### TensorFrame (tensorframe.frame)

**Construcción:**
- `TensorFrame(data, indices, attrs)` — constructor principal
- `tf.field(data, dims, dtype, name)` — helper para crear campos
- `tf.tensor_field(data, dims, name)` — helper para campos tensoriales

**Propiedades:** `schema`, `field_names`, `dims`, `indices`, `shape`, `num_fields`, `attrs`

**Acceso:** `frame["campo"]`, `frame[["c1","c2"]]`, `frame.get_array("campo")`

**Indexing:** `isel(**dims)`, `sel(**dims)`, `where(mask, dim)`

**Transformación:** `with_column()`, `drop_fields()`, `rename_dims()`, `apply(fn, fields)`

**Operaciones:** `groupby(by)`, `map(fn, dim)`

**ML Pipeline:** `dropna()`, `fillna()`, `normalize()`, `encode_categorical()`, `one_hot()`, `split()`, `to_jax_arrays()`, `iter_batches()`

**Storage:** `save(path, chunk_config)`

**Conversión:** `to_dict()`, `to_numpy()`

### Funciones de Módulo

| Función | Módulo | Descripción |
|---|---|---|
| `tf.concat(frames, dim)` | ops | Concatenar frames |
| `tf.merge(left, right, on, how)` | ops | Merge/join |
| `tf.save(frame, path)` | storage | Guardar a Zarr |
| `tf.open(path, lazy)` | storage | Abrir desde Zarr |
| `tf.register_kernel(name, fn)` | ops | Registrar kernel personalizado |
| `tf.get_kernel(name)` | ops | Obtener kernel registrado |

### Excepciones

```
TensorFrameError
├── SchemaError
│   ├── SchemaValidationError
│   ├── SchemaMismatchError
│   └── SchemaEvolutionError
├── StorageError
│   ├── MaterializationError
│   ├── DeviceMemoryError
│   └── PersistenceError
├── ComputeError
│   ├── ShapeError
│   ├── DtypeError
│   └── JITTraceError
├── IndexLabelError
└── DimensionError
```

---

## 7. Testing y Calidad

### Ejecución de Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Con cobertura
python -m pytest tests/ --cov=tensorframe --cov-report=term-missing

# Tests de un módulo específico
python -m pytest tests/test_frame.py -v
```

### Distribución de Tests

| Suite | Tests | Cobertura del Módulo |
|---|---|---|
| test_ndtype.py | 65 | 93% |
| test_schema.py | 26 | 98% |
| test_index.py | 26 | 98% |
| test_errors.py | 7 | 100% |
| test_construction.py | 11 | 100% |
| test_frame.py | 57 | 98% |
| test_series.py | 8 | 100% |
| test_storage.py | 20 | 87% |
| test_ops.py | 30 | 87% |
| test_ml.py | 46 | 91% |
| **Total** | **296** | **93%** |

### Categorías de Tests

- **Unitarios:** Tipos, esquemas, índices individuales
- **Integración:** TensorFrame con operaciones completas, pytree round-trips, save/open cycles
- **JAX Compatibility:** JIT, vmap, tree_map, pytree flatten/unflatten
- **Pipeline:** Flujos completos de ML (normalize → split → extract → batch)
- **Edge Cases:** NaN, empty frames, missing keys, conflicting schemas

---

## 8. Guía de Contribución

### Principios de Diseño

1. **Inmutabilidad:** Toda operación retorna un nuevo objeto. Nunca mutar in-place.
2. **Pytree-first:** Nuevas estructuras deben registrarse como pytrees JAX.
3. **Tipos explícitos:** Usar NDType para toda representación de tipos.
4. **Errores descriptivos:** Usar la jerarquía de excepciones existente con mensajes claros.
5. **Funcional:** Preferir funciones puras. Los métodos de TensorFrame delegan a módulos.

### Agregar una Nueva Operación

1. Implementar la función en el módulo correspondiente (`ops.py`, `ml.py`).
2. Agregar método delegado en `TensorFrame` (en `frame.py`).
3. Exportar en `__init__.py`.
4. Escribir tests que cubran: caso normal, edge cases, errores esperados.
5. Ejecutar `pytest --cov` y verificar cobertura > 90%.

### Convenciones

- Docstrings en formato NumPy
- Type hints en todas las firmas públicas
- Tests en `tests/test_<módulo>.py`
- Funciones puras donde sea posible
