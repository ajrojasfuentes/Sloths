# TensorFrame: Documento de Diseño Arquitectónico

## Framework NDFrame Multidimensional sobre JAX con Storage Layer Unificado

**Versión:** 0.1-draft  
**Fecha:** Marzo 2026  
**Estado:** Propuesta Arquitectónica (sin código)

---

## 1. Resumen Ejecutivo

TensorFrame es un framework de estructuras de datos etiquetadas y multidimensionales que reemplaza el backend NumPy de Pandas por JAX, permitiendo cómputo acelerado en GPU/TPU con paralelismo automático. El framework incorpora un Storage Layer de tres capas (JAX Array → TensorStore → Zarr v3) que actúa como análogo de Apache Arrow pero para datos N-dimensionales y anidados: ofrece un formato en memoria estandarizado optimizado para cómputo acelerado, I/O asíncrono, caching inteligente y persistencia chunked/comprimida.

El caso de uso principal es servir como herramienta de propósito general para análisis científico, limpieza/normalización de datasets, y preparación de datos previo a su consumo por frameworks de ML/DL.

---

## 2. Análisis de Componentes Base

### 2.1 NDFrame de Pandas: Lecciones Aprendidas

El `NDFrame` de Pandas es la clase base abstracta de `Series` (1D) y `DataFrame` (2D). Su arquitectura interna se articula alrededor de varios componentes que debemos entender para decidir qué conservar, qué rediseñar y qué descartar.

#### 2.1.1 BlockManager (lo que reemplazamos)

El `BlockManager` es el corazón del almacenamiento interno de Pandas. Fue introducido en 2011 por Wes McKinney para superar las limitaciones del diseño original basado en `dict`. Su función es agrupar columnas del mismo dtype en bloques contiguos de memoria NumPy, permitiendo operaciones eficientes sobre columnas homogéneas sin iterar columna por columna.

**Funcionamiento:** Cuando se crea un DataFrame, Pandas consolida las columnas del mismo tipo en bloques NumPy 2D compartidos. Por ejemplo, un DataFrame con 3 columnas `int64` y 2 `float64` producirá 2 bloques internos (uno `int64` de shape `(3, N)` y uno `float64` de shape `(2, N)`). Cada columna individual es una vista (view) sobre su bloque, lo que permite operaciones como `sum` por columna sin copiar datos.

**Problemas identificados:** La consolidación agrega complejidad: no ocurre en el momento de inserción sino bajo demanda en ciertas operaciones (como `diff`, `fillna`, `reindex`), lo que hace impredecible el rendimiento. Agregar columnas una a una fragmenta los bloques, degradando operaciones posteriores. Además, el BlockManager tiene un efecto de duplicación de memoria: operaciones de consolidación pueden requerir copiar todos los datos temporalmente. El propio Wes McKinney ha documentado que el sistema necesita 5-10x la RAM del tamaño del dataset.

**Decisión de diseño para TensorFrame:** No replicaremos el BlockManager. En su lugar, usaremos un modelo de "un campo = un array JAX" sin consolidación por dtype, similar a lo que hace Arrow internamente con buffers independientes por columna. La razón es que JAX ya maneja la memoria de forma optimizada para el hardware subyacente, y la consolidación por dtype pierde sentido cuando el backend computacional (XLA) realiza sus propias fusiones de operaciones a nivel de compilación.

#### 2.1.2 Index y Ejes Etiquetados (lo que conservamos y extendemos)

El sistema de ejes etiquetados es el aspecto más valioso de Pandas. Cada eje de un NDFrame tiene un `Index` asociado que permite: alineación automática entre estructuras con distintos índices, selección por etiquetas (`.loc`), selección posicional (`.iloc`), y MultiIndex para indexación jerárquica.

**Decisión:** Conservaremos ejes etiquetados y los generalizaremos a N dimensiones. Cada dimensión tendrá un nombre (`dim_name`) y un `Index` asociado. Esto es conceptualmente similar a lo que hace Xarray con sus coordenadas, pero integrado directamente en la estructura base del framework.

#### 2.1.3 Copy-on-Write (lo que adoptamos como principio)

Pandas 3.0 adoptó Copy-on-Write (CoW) como comportamiento por defecto. Bajo CoW, las vistas y subconjuntos de datos comparten memoria hasta que uno de ellos es modificado, momento en el cual se realiza una copia defensiva.

**Decisión:** En TensorFrame, CoW es innecesario como mecanismo porque JAX ya impone inmutabilidad funcional: los arrays JAX no se modifican in-place. Cada operación produce un array nuevo. Esto es más limpio que CoW y es requerido por las transformaciones de JAX (`jit`, `grad`, `vmap`). Adoptaremos la inmutabilidad total como principio de diseño, con APIs funcionales del tipo `frame.set_column("col", new_data)` que retornan un nuevo frame.

#### 2.1.4 ExtensionDtype/ExtensionArray (lo que reinventamos)

Pandas permite tipos personalizados mediante `ExtensionDtype` y `ExtensionArray`. En la práctica, es un sistema bolt-on que no integra bien con el BlockManager original.

**Decisión:** Diseñaremos un sistema de tipos propio basado en las capacidades de JAX y alineado con la especificación Zarr v3 para persistencia. El sistema soportará tipos escalares (int, float, bool, complex, datetime), tipos anidados (listas de longitud variable, structs, unions) y tipos tensoriales (campos que contienen sub-tensores de forma fija).

### 2.2 JAX: Modelo Computacional y Restricciones

#### 2.2.1 Transformaciones Funcionales

JAX opera sobre funciones puras. Sus cuatro transformaciones centrales imponen restricciones que nuestro diseño debe respetar:

- **`jit` (Just-In-Time Compilation):** Compila funciones a XLA para ejecución optimizada en CPU/GPU/TPU. Requiere que las formas (shapes) de los arrays sean estáticas y conocidas en tiempo de compilación. No admite control de flujo Python que dependa de valores concretos de arrays. Esto significa que la API de TensorFrame no puede usar condicionales basados en datos dentro de operaciones `jit`-compiladas; debemos usar `jax.lax.cond`, `jax.lax.scan`, etc.
- **`grad` (Diferenciación Automática):** Calcula gradientes de funciones escalares. Requiere funciones puras sin efectos secundarios. Esto es relevante porque los usuarios querrán aplicar `grad` sobre pipelines que incluyan operaciones de TensorFrame (por ejemplo, normalización como paso diferenciable).
- **`vmap` (Vectorización Automática):** Aplica una función sobre un eje batch sin escribir bucles explícitos. Es esencial para procesar batches de datos. Nuestras estructuras deben ser compatibles con `vmap`, lo que requiere que los TensorFrames sean pytrees válidos.
- **`pmap`/`shard_map` (Paralelismo Multi-dispositivo):** Distribuye cómputo entre múltiples GPUs/TPUs. Los arrays se pueden shardar automáticamente entre dispositivos.

#### 2.2.2 Pytrees: La Clave de la Integración

JAX opera sobre "pytrees": cualquier estructura anidada de contenedores Python (listas, tuplas, dicts) cuyas hojas son arrays JAX. Las transformaciones de JAX (`jit`, `grad`, `vmap`) atraviesan pytrees automáticamente, aplicando operaciones a cada hoja-array.

**Implicación crítica:** Para que TensorFrame sea un ciudadano de primera clase en JAX, sus estructuras deben registrarse como pytrees personalizados. Esto significa implementar funciones `tree_flatten` y `tree_unflatten` que indiquen a JAX cómo descomponer un TensorFrame en sus arrays constituyentes (hojas) y sus metadatos (estructura del árbol), y cómo recomponerlo después de una transformación.

El pytree de un TensorFrame separa:
- **Hojas (datos dinámicos):** los arrays JAX que contienen los datos de cada campo/columna.
- **Aux data (metadatos estáticos):** nombres de campos, nombres de dimensiones, índices, esquema de tipos, y forma de la estructura.

Esta separación es fundamental porque `jit` solo rastrea (trace) las hojas, mientras que los metadatos se tratan como constantes de compilación.

#### 2.2.3 Inmutabilidad y Pattern de Actualización

Los arrays JAX son inmutables: no existe `array[i] = value`. En su lugar, JAX provee `array.at[i].set(value)`, que retorna un nuevo array. Nuestras estructuras seguirán este patrón funcional:

```
# Conceptual — no ejecutable
new_frame = frame.with_column("precio", frame["precio"] * 1.16)
new_frame = frame.rename_dim("batch", "sample")
new_frame = frame.reindex("tiempo", new_time_index)
```

Cada método retorna una nueva instancia. Internamente, JAX/XLA puede optimizar estas operaciones para evitar copias innecesarias durante la compilación JIT.

#### 2.2.4 Sharding y Paralelismo

`jax.Array` es el tipo unificado que puede representar arrays en un solo dispositivo, shardados entre múltiples dispositivos, o distribuidos entre múltiples hosts. El sharding se especifica mediante `NamedSharding(mesh, PartitionSpec(...))`.

TensorFrame propagará la información de sharding a nivel de campo: cada campo de un TensorFrame puede tener su propia estrategia de sharding, y las operaciones que combinen campos respetarán las restricciones de compatibilidad de sharding de JAX.

### 2.3 Apache Arrow: Analogías para el Mundo ND

Arrow es el estándar de facto para datos columnar en memoria, optimizado para datos tabulares (2D). Sus principios de diseño que adoptaremos como analogía para datos ND son:

#### 2.3.1 Formato en Memoria Estandarizado

Arrow define un layout físico en memoria que permite zero-copy entre procesos y lenguajes. Los buffers están alineados a 64 bytes para SIMD, cada array tiene un bitmap de validez (nulls), y los tipos anidados (List, Struct, Union) se descomponen en buffers planos de offsets + datos.

**Analogía TensorFrame:** Definiremos un "TensorFrame Memory Layout" estandarizado: cada campo se almacena como un `jax.Array` cuyo buffer subyacente está alineado y gestionado por XLA. Los metadatos (esquema, índices, nombres de dimensiones) se serializan en FlatBuffers o MessagePack para IPC zero-copy. Los campos anidados se representan como pytrees de arrays, donde offsets y datos viven en buffers separados igual que en Arrow.

#### 2.3.2 RecordBatch como Unidad de Transporte

En Arrow, el `RecordBatch` es la unidad atómica de serialización: contiene un esquema y los buffers de datos correspondientes. Se puede deserializar sin copiar memoria.

**Analogía TensorFrame:** Nuestra unidad equivalente será el `TensorBatch`: un snapshot serializable de un TensorFrame (o slice del mismo) que incluye los arrays JAX materializados + esquema + índices. Los TensorBatch se podrán transmitir entre procesos o materializar desde/hacia TensorStore/Zarr sin serialización completa.

#### 2.3.3 Tipos Anidados

Arrow soporta List, Struct, Map, Union con buffers planos de offsets. Esto permite representar JSON-like data en formato columnar.

**Analogía TensorFrame:** Extenderemos este concepto a N dimensiones. Un campo puede contener: un tensor de forma fija (e.g., una imagen `(H, W, C)` por fila), una lista de longitud variable de tensores (usando offsets como Arrow), o un struct de sub-campos donde cada sub-campo es a su vez un tensor ND.

### 2.4 TensorStore: Motor de I/O Asíncrono

TensorStore es una librería de Google para lectura/escritura eficiente de arrays multidimensionales masivos. Sus características clave para nuestro diseño:

- **API Asíncrona:** Todas las operaciones retornan `Future` objects. Las lecturas y escrituras se ejecutan en background mientras el programa continúa. Esto es crucial para pipelines donde queremos solapar I/O con cómputo GPU.
- **Caching en Memoria:** Configurable con `cache_pool`, reduce accesos a storage subyacente para datos frecuentemente accedidos.
- **Vistas Virtuales:** Las operaciones de indexing producen vistas sin copiar datos. Los `IndexTransform` son composables: se pueden apilar slicing, transposición y broadcasting como transformaciones lazy.
- **Drivers Múltiples:** Soporta zarr, N5, Neuroglancer precomputed, con backends de storage variados (local, GCS, S3, in-memory).
- **Concurrencia Optimista:** Acceso seguro desde múltiples procesos/máquinas mediante optimistic concurrency, con garantías ACID por operación.
- **Integración con JAX:** Ya usado por proyectos como PaLM y T5X para checkpointing de modelos distribuidos.

### 2.5 Zarr v3: Persistencia Chunked/Comprimida

Zarr v3 es la especificación de formato para almacenamiento de arrays ND chunked y comprimidos. Características relevantes:

- **Codec Pipeline:** Zarr v3 define una pipeline de codecs composable: `array → array` (filtros como Delta), `array → bytes` (serialización como BytesCodec), y `bytes → bytes` (compresión como ZstdCodec, BloscCodec). Esta modularidad permite optimizar la pipeline por tipo de dato.
- **Sharding:** En v3, múltiples chunks pueden almacenarse en un solo objeto de storage (shard). Los shards son la unidad de escritura; los chunks individuales son la unidad de lectura. Esto reduce drásticamente el número de objetos en storage para arrays con chunks pequeños.
- **Store Abstraction:** Define un interfaz abstracto de key-value store, permitiendo backends variados (filesystem, S3, memory).
- **Core Asíncrono:** Zarr-Python 3 es internamente asíncrono con `asyncio`, despachando operaciones de compresión a thread pools.

---

## 3. Arquitectura Propuesta

### 3.1 Visión General de Capas

```
┌─────────────────────────────────────────────────────────────────┐
│                        Capa de Usuario                          │
│  TensorFrame  ·  TensorSeries  ·  TensorGroup  ·  Pipeline API │
│  (API etiquetada, indexing, groupby, merge, reshape, apply)     │
├─────────────────────────────────────────────────────────────────┤
│                     Capa de Esquema y Tipos                     │
│  NDSchema  ·  NDType  ·  Index  ·  MultiIndex  ·  DimAxis       │
│  (Definición de estructura, validación, metadatos)              │
├─────────────────────────────────────────────────────────────────┤
│                    Capa de Cómputo (JAX)                        │
│  Kernel Registry  ·  LazyExpr Graph  ·  JIT Bridge              │
│  (Evaluación lazy, compilación, vmap/grad/pmap hooks)           │
├─────────────────────────────────────────────────────────────────┤
│                 Capa de Storage (3 niveles)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Nivel 1: JAX Array (Hot)                                 │  │
│  │  Arrays materializados en memoria de dispositivo           │  │
│  │  (GPU/TPU/CPU). Listos para cómputo inmediato.            │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  Nivel 2: TensorStore (Warm)                              │  │
│  │  Vistas virtuales, cache L2, I/O asíncrono,               │  │
│  │  composición de IndexTransform, concurrencia.             │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  Nivel 3: Zarr v3 (Cold)                                  │  │
│  │  Persistencia chunked/comprimida en disco, cloud storage, │  │
│  │  o cualquier KV-store. Sharding para escritura eficiente. │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Capa de Interop                                │
│  Arrow Bridge  ·  NumPy Bridge  ·  DLPack  ·  Parquet/CSV I/O  │
│  (Conversión zero-copy donde sea posible)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Estructuras de Datos Principales

#### 3.2.1 `NDSchema`: Esquema Tipado N-Dimensional

El NDSchema es el descriptor completo de la estructura de un TensorFrame. Juega el mismo rol que `Schema` en Arrow pero para datos ND:

```
NDSchema
├── fields: OrderedDict[str, FieldSpec]
│   ├── FieldSpec
│   │   ├── name: str
│   │   ├── dtype: NDType
│   │   ├── dims: tuple[str, ...]       # nombres de dimensiones del campo
│   │   ├── shape: tuple[int|None, ...]  # None = dimensión variable
│   │   ├── nullable: bool
│   │   └── metadata: dict[str, Any]
├── dims: OrderedDict[str, DimSpec]
│   ├── DimSpec
│   │   ├── name: str
│   │   ├── size: int | None            # None = dinámico
│   │   ├── index: Index | None
│   │   └── coord_fields: list[str]     # campos que actúan como coordenadas
└── metadata: dict[str, Any]
```

**Sistema de Tipos (`NDType`):**

| Categoría | Tipos | Notas |
|-----------|-------|-------|
| Escalares | `bool_`, `int8..64`, `uint8..64`, `float16..64`, `bfloat16`, `complex64/128` | Mapeo directo a dtypes JAX |
| Temporal | `datetime64[unit]`, `timedelta64[unit]` | Almacenados como int64 internamente |
| Cadenas | `string` (UTF-8 variable), `fixed_string[N]` | Offsets + buffer de bytes (como Arrow) |
| Tensor | `tensor[dtype, shape]` | Campo que contiene un sub-tensor de forma fija por elemento |
| Lista | `list_[inner_type]`, `fixed_list[inner_type, N]` | Variable: offsets + valores. Fija: reshape |
| Struct | `struct[{name: type, ...}]` | Cada sub-campo es un array independiente |
| Nullable | `nullable[inner_type]` | Array de datos + bitmap de validez |
| Categorical | `categorical[categories, ordered]` | Indices enteros + tabla de categorías |

El tipo `tensor[float32, (224, 224, 3)]` es particularmente importante: permite que cada "fila" de un TensorFrame contenga una imagen completa, almacenada como un array JAX contiguo de forma `(N, 224, 224, 3)` donde `N` es el número de filas.

#### 3.2.2 `TensorFrame`: Estructura Principal

Un TensorFrame es la estructura central del framework. Conceptualmente es un contenedor etiquetado de campos N-dimensionales que comparten una o más dimensiones comunes (los "ejes de alineación").

**Anatomía interna:**

```
TensorFrame
├── _schema: NDSchema              # Descriptor de estructura (estático para JAX)
├── _fields: dict[str, FieldData]  # Datos por campo
│   └── FieldData
│       ├── data: jax.Array | StorageRef  # Array materializado O referencia lazy
│       ├── mask: jax.Array | None        # Bitmap de nulls (si nullable)
│       └── state: Literal["hot", "warm", "cold"]
├── _indices: dict[str, Index]     # Un Index por dimensión compartida
├── _dim_order: tuple[str, ...]    # Orden de dimensiones compartidas
└── _attrs: dict[str, Any]         # Metadatos de usuario
```

**Invariantes fundamentales:**

1. Todos los campos comparten al menos una dimensión ("eje de alineación primario", típicamente la primera dimensión). Esa dimensión tiene el mismo tamaño en todos los campos.
2. Los campos pueden tener dimensiones adicionales propias. Por ejemplo, un campo `imagen` puede tener dims `("batch", "alto", "ancho", "canal")` mientras que `etiqueta` tiene solo `("batch",)`.
3. El esquema es inmutable. Cambiar la estructura (agregar/quitar campos, cambiar tipos) produce un nuevo TensorFrame con nuevo esquema.
4. Los datos son inmutables (requisito de JAX). Toda operación retorna un nuevo TensorFrame.

**Registro como Pytree:**

```python
# Conceptual
def tree_flatten(frame):
    # Hojas: todos los arrays de datos + masks (dinámicos, rastreados por JAX)
    leaves = []
    for name in frame._schema.field_names:
        leaves.append(frame._fields[name].data)
        if frame._fields[name].mask is not None:
            leaves.append(frame._fields[name].mask)
    
    # Aux data: todo lo demás (estático, constante para JAX)
    aux = (frame._schema, frame._indices, frame._dim_order, frame._attrs)
    return leaves, aux

def tree_unflatten(aux, leaves):
    schema, indices, dim_order, attrs = aux
    # Reconstruir TensorFrame desde hojas transformadas + metadatos preservados
    ...
```

Esto permite:
- `jax.jit(lambda f: f["precio"] * 2)(mi_frame)` → compilación JIT
- `jax.vmap(lambda f: f["imagen"].mean())(batch_frames)` → vectorización sobre batch
- `jax.grad(loss_fn)(frame_params)` → gradientes a través de operaciones de TensorFrame

#### 3.2.3 `TensorSeries`: Caso Especial 1D

Un TensorSeries es un TensorFrame con un solo campo. Es el análogo de `Series` en Pandas. Internamente, es simplemente un TensorFrame con la restricción de tener exactamente un campo, más azúcar sintáctico para acceso directo a los datos.

#### 3.2.4 `TensorGroup`: Contenedor Jerárquico

Un TensorGroup es un contenedor que agrupa múltiples TensorFrames y/o otros TensorGroups en una jerarquía con nombre, análogo a los `Group` de Zarr/HDF5. Permite organizar datasets complejos:

```
dataset (TensorGroup)
├── train (TensorFrame)  — campos: imagen, etiqueta, metadatos
├── test (TensorFrame)   — campos: imagen, etiqueta
└── normalization (TensorGroup)
    ├── means (TensorSeries)
    └── stds (TensorSeries)
```

### 3.3 Storage Layer: Detalle de las 3 Capas

#### 3.3.1 Nivel 1 — JAX Array (Hot Storage)

**Propósito:** Datos materializados en memoria de dispositivo, listos para cómputo inmediato.

**Características:**
- Tipo subyacente: `jax.Array`, que unifica `DeviceArray`, `ShardedDeviceArray` y `GlobalDeviceArray`.
- Puede residir en CPU, GPU o TPU. Si hay múltiples dispositivos, puede estar shardado con `NamedSharding`.
- Inmutable. Toda operación produce un nuevo array.
- Soporta todas las transformaciones JAX: `jit`, `grad`, `vmap`, `shard_map`.

**Cuándo se usa:** Cuando los datos caben en memoria del dispositivo y se van a computar inmediatamente. Es el estado por defecto para TensorFrames pequeños/medianos.

**Materialización:** Los datos en niveles Warm/Cold se materializan a Hot mediante:
```
StorageRef(tensorstore_spec) → await ts.open(spec) → ts[slice].read() → jax.Array
```

#### 3.3.2 Nivel 2 — TensorStore (Warm Storage)

**Propósito:** Capa intermedia que provee I/O asíncrono, caching, vistas virtuales y acceso concurrente.

**Características:**
- Los campos que no están materializados en GPU mantienen un `StorageRef` que apunta a un TensorStore abierto.
- El cache en memoria de TensorStore (`cache_pool`) actúa como L2 cache: los chunks accedidos recientemente se mantienen en RAM del host sin ocupar memoria de dispositivo.
- Las operaciones de indexing sobre campos warm se traducen a `IndexTransform` de TensorStore, que son composables y lazy: no leen datos hasta que se materializa.
- Las escrituras usan write-back caching con flush controlado.

**Cuándo se usa:** Para campos de datasets que no caben en memoria de dispositivo, o para acceso parcial a datasets grandes. También como capa de staging al escribir resultados a disco.

**Configuración de contexto:**
```
Context:
  cache_pool: 2GB              # Cache L2 en RAM del host
  data_copy_concurrency: 8     # Hilos para encoding/decoding
  file_io_concurrency: 32      # Operaciones I/O concurrentes
```

#### 3.3.3 Nivel 3 — Zarr v3 (Cold Storage)

**Propósito:** Persistencia durable, chunked y comprimida en disco o cloud storage.

**Características:**
- Cada campo de un TensorFrame se persiste como un `zarr.Array` dentro de un `zarr.Group`.
- Los metadatos del TensorFrame (esquema, índices, atributos) se almacenan como atributos JSON del grupo raíz.
- Chunking configurable por campo: un campo de imágenes `(N, 224, 224, 3)` podría chunkearse como `(64, 224, 224, 3)` (un chunk = 64 imágenes completas).
- Sharding de Zarr v3: múltiples chunks en un solo archivo para reducir overhead de filesystem.
- Pipeline de codecs por campo: por ejemplo, imágenes con `BytesCodec + BloscCodec(cname='zstd')`, datos numéricos con `Delta + BytesCodec + ZstdCodec`.

**Layout en disco:**
```
dataset.tensorframe.zarr/
├── zarr.json                    # Metadatos del grupo raíz + esquema TensorFrame
├── imagen/
│   ├── zarr.json                # Metadatos del array (shape, chunks, codecs)
│   └── c/0/0/0/0               # Chunks de datos (o shards)
├── etiqueta/
│   ├── zarr.json
│   └── c/0
├── _indices/
│   ├── batch/zarr.json          # Index de la dimensión "batch"
│   └── batch/c/0
└── _schema.json                 # NDSchema serializado
```

#### 3.3.4 Flujo entre Niveles y Políticas de Promoción/Evicción

```
              MATERIALIZAR                    CARGAR
Cold (Zarr) ──────────────→ Warm (TensorStore) ──────────→ Hot (JAX Array)
              ←──────────────                  ←────────────
               PERSISTIR                        EVICTAR

Políticas:
  - Auto-materialize: campos accedidos para cómputo se promueven automáticamente
                      de Warm → Hot justo antes de la operación.
  - Prefetch:         durante iteración (e.g., DataLoader), se pre-cargan los
                      siguientes N batches asincrónicamente.
  - Eviction LRU:     campos Hot no accedidos recientemente se degradan a Warm
                      cuando la memoria del dispositivo se llena.
  - Write-through:    las escrituras van a Hot y se propagan asincrónicamente
                      a Warm → Cold.
  - Lazy-by-default:  al abrir un dataset desde disco, TODO empieza en Cold.
                      Solo se materializa lo que se usa.
```

### 3.4 Capa de Cómputo

#### 3.4.1 Evaluación Lazy con Grafo de Expresiones

Las operaciones sobre TensorFrames no se ejecutan inmediatamente. En su lugar, construyen un grafo de expresiones (`LazyExpr`) que se evalúa cuando:
1. El usuario solicita materialización explícita (`.compute()`, `.to_jax()`).
2. Se necesita un valor concreto (impresión, conversión a NumPy, escritura a disco).
3. Una función decorada con `@jax.jit` necesita los datos como input.

```
LazyExpr Graph:
  frame["precio"] * 1.16 + frame["impuesto"]
  
  Se representa como:
  Add(
    Mul(FieldRef("precio"), Scalar(1.16)),
    FieldRef("impuesto")
  )
  
  Al evaluar, se traduce a operaciones JAX:
  jnp.add(jnp.multiply(fields["precio"], 1.16), fields["impuesto"])
```

La evaluación lazy permite:
- Fusión de operaciones: `(x * 2 + 1).sum()` se compila como una sola expresión XLA.
- Eliminación de materializaciones intermedias.
- Despacho optimizado: si los campos están en Cold, se pueden cargar solo los chunks necesarios.

#### 3.4.2 Kernel Registry

Las operaciones primitivas (suma, media, agrupación, merge, etc.) se implementan como "kernels" registrados en un registry central. Cada kernel tiene:
- Una implementación JAX pura (función que opera sobre `jax.Array`).
- Reglas de propagación de esquema (cómo cambia el esquema del output respecto al input).
- Reglas de propagación de sharding (cómo se redistribuyen los datos si están shardados).
- Opcionalmente, una versión `jit`-compilada pre-cacheada.

#### 3.4.3 Bridge con Transformaciones JAX

Para que las operaciones de TensorFrame sean compatibles con `jit`/`grad`/`vmap`, exponemos un API funcional puro:

```python
# Funciones que aceptan y retornan TensorFrames (pytrees)
@jax.jit
def normalizar(frame):
    media = frame["valores"].mean(dim="tiempo")
    std = frame["valores"].std(dim="tiempo")
    return frame.with_column("valores_norm", (frame["valores"] - media) / std)

@jax.vmap  # vectoriza sobre primera dimensión
def procesar_batch(frame):
    return frame.apply(jnp.fft.fft, fields=["señal"])

@jax.grad
def loss(frame):
    pred = modelo(frame["features"])
    return jnp.mean((pred - frame["target"]) ** 2)
```

### 3.5 API de Usuario

#### 3.5.1 Construcción

```python
# Desde datos en memoria
frame = tf.TensorFrame({
    "imagen": tf.tensor_field(jnp.zeros((1000, 224, 224, 3)), dims=("batch", "h", "w", "c")),
    "etiqueta": tf.field(jnp.array([0, 1, 2, ...]), dims=("batch",)),
    "timestamp": tf.field(timestamps, dims=("batch",), dtype=tf.datetime64["s"]),
})

# Desde disco (lazy, nada se carga en memoria)
frame = tf.open("dataset.tensorframe.zarr")

# Desde Arrow Table (zero-copy donde sea posible)
frame = tf.from_arrow(arrow_table)

# Desde Pandas DataFrame
frame = tf.from_pandas(df)
```

#### 3.5.2 Indexing y Selección

```python
# Por etiqueta (como .loc)
frame.sel(batch="sample_42", h=slice(0, 100))

# Por posición (como .iloc)
frame.isel(batch=42, h=slice(0, 100))

# Por campo
frame["imagen"]        # TensorSeries
frame[["imagen", "etiqueta"]]  # TensorFrame con subset de campos

# Boolean indexing
frame.where(frame["etiqueta"] == 1)

# Fancy indexing sobre dimensión
frame.sel(batch=["s1", "s2", "s10"])
```

#### 3.5.3 Operaciones de Transformación

```python
# Renombrar dimensiones
frame.rename_dims({"batch": "sample"})

# Agregar/reemplazar campo (retorna nuevo frame)
frame.with_column("precio_iva", frame["precio"] * 1.16)

# Drop
frame.drop_fields(["temp_col"])

# Reshape / transponer
frame.transpose("c", "h", "w", "batch")

# GroupBy (inspirado en Pandas pero para ND)
frame.groupby("categoria").agg({"precio": "mean", "cantidad": "sum"})

# Merge/Join (por índice o por campo)
tf.merge(frame_a, frame_b, on="id", how="inner")

# Concatenación
tf.concat([frame_1, frame_2], dim="batch")

# Apply (aplica función JAX a campos seleccionados)
frame.apply(jnp.log1p, fields=["valores"])

# Map sobre dimensión (como vmap conceptual)
frame.map(lambda row: row["imagen"].mean(), dim="batch")
```

#### 3.5.4 Operaciones de Cleaning/Normalización (Caso de Uso ML)

```python
# Detección y manejo de nulls
frame.dropna(dim="batch")
frame.fillna({"precio": 0.0, "nombre": "<unknown>"})

# Normalización estadística
frame.normalize(fields=["f1", "f2"], method="zscore", dim="batch")
frame.normalize(fields=["imagen"], method="minmax", dim=("h", "w"))

# Encoding categórico
frame.encode_categorical("color", categories=["rojo", "verde", "azul"])
frame.one_hot("color")

# Split para ML
train, val, test = frame.split(dim="batch", ratios=[0.7, 0.15, 0.15], shuffle=True, seed=42)

# Conversión a tensores puros para frameworks ML
X, y = frame.to_jax_arrays(features=["f1", "f2", "f3"], target="label")

# Batching con DataLoader integrado
for batch in frame.iter_batches(batch_size=64, shuffle=True, prefetch=2):
    # batch es un TensorFrame con 64 elementos, ya materializado en GPU
    ...
```

#### 3.5.5 Persistencia

```python
# Guardar a Zarr v3
frame.save("output.tensorframe.zarr",
    chunk_config={
        "imagen": {"chunks": (64, 224, 224, 3), "codecs": ["blosc_zstd"]},
        "etiqueta": {"chunks": (1024,), "codecs": ["zstd"]},
    }
)

# Exportar a formatos conocidos
frame.to_parquet("output.parquet")  # Solo campos 1D/2D
frame.to_arrow()                     # Arrow Table
frame.to_pandas()                    # Pandas DataFrame (solo 2D)
frame.to_csv("output.csv")          # Solo campos escalares
```

---

## 4. Decisiones de Diseño Clave

### 4.1 Inmutabilidad Total vs. Mutabilidad Controlada

**Decisión: Inmutabilidad total.** Todos los TensorFrame y sus campos son inmutables una vez creados. Las operaciones retornan nuevos objetos.

**Justificación:** Es un requisito duro de JAX para `jit`, `grad` y `vmap`. Además, elimina bugs de aliasing, simplifica el razonamiento sobre el código, y permite optimizaciones como structural sharing (los arrays JAX no copiados se comparten por referencia).

**Trade-off:** Los usuarios acostumbrados a `df["col"] = valor` deberán adaptarse al estilo funcional. Proveeremos una API "builder" para construcción progresiva que internamente acumula cambios y construye el frame al final.

### 4.2 Lazy-by-Default vs. Eager

**Decisión: Eager para datos en memoria, Lazy para I/O.**

Las operaciones aritméticas sobre arrays ya materializados se ejecutan inmediatamente (como JAX estándar). Pero la carga de datos desde disco es siempre lazy: abrir un dataset Zarr no lee ningún dato; los chunks se cargan bajo demanda.

**Justificación:** JAX ya es internamente async (despacho asíncrono), así que hacerlo "lazy" a nivel de operaciones aritméticas agrega complejidad sin beneficio claro para la mayoría de los casos. Sin embargo, para I/O, lazy es esencial porque los datasets pueden ser mucho más grandes que la memoria disponible.

### 4.3 Un Campo = Un Array vs. BlockManager

**Decisión: Un campo = un array JAX independiente.** No hay consolidación por dtype.

**Justificación:** XLA (el compilador detrás de JAX) ya fusiona operaciones sobre múltiples arrays cuando están dentro de una función `jit`. La consolidación por dtype del BlockManager era necesaria porque NumPy no tenía esta capacidad. Con JAX/XLA, la fusión de operaciones ocurre a nivel de compilación, no a nivel de layout de memoria.

### 4.4 Forma de los Campos: Homogénea vs. Heterogénea

**Decisión: Cada campo tiene su propia forma, pero todos comparten al menos una dimensión de alineación.**

Esto permite que un TensorFrame tenga campos de distintas dimensionalidades (un escalar, un vector, una imagen por fila), siempre que compartan la dimensión de "batch" o "registro". Es más flexible que un DataFrame (donde todo es 1D por columna) pero más estructurado que un pytree libre.

### 4.5 Manejo de Datos de Longitud Variable

**Decisión: Offsets + valores planos (como Arrow), con soporte para padding + mask como alternativa.**

Para campos de tipo `list_[T]` donde cada elemento puede tener distinta longitud (por ejemplo, secuencias de texto tokenizadas), almacenamos:
- Un array de offsets `int32/int64` de shape `(N+1,)`.
- Un array de valores planos de shape `(total_elements,)`.
- Elemento `i` tiene valores `values[offsets[i]:offsets[i+1]]`.

Alternativamente, para uso con JAX (que requiere formas estáticas), ofrecemos conversión a "padded + mask": arrays rectangulares con padding a la longitud máxima y un mask booleano.

---

## 5. Consideraciones de Performance

### 5.1 Materialización Just-In-Time

Cuando una operación `jit`-compilada necesita datos de un campo que está en Cold/Warm, se materializan asincrónicamente al dispositivo antes de la ejecución. El grafo de ejecución coordina:

```
1. Analizar qué campos necesita la función JIT
2. En paralelo: fetch de chunks necesarios vía TensorStore (async)
3. Transfer host → device (DMA, solapado con fetch)
4. Ejecutar kernel XLA compilado
5. Si resultado debe persistir: write-back asíncrono
```

### 5.2 Prefetching para Iteración

El `iter_batches()` implementa un pipeline de doble buffer:
- Mientras el batch actual se procesa en GPU, el siguiente batch se carga asincrónicamente desde TensorStore.
- Se usa `jax.device_put` con `donate_argnums` para reciclar buffers de device.

### 5.3 Sharding Automático

Para datasets que no caben en un solo dispositivo, TensorFrame propaga hints de sharding:

```python
mesh = jax.sharding.Mesh(jax.devices(), ("data",))
sharding = tf.ShardingConfig({
    "imagen": NamedSharding(mesh, P("data", None, None, None)),
    "etiqueta": NamedSharding(mesh, P("data",)),
})
frame = frame.with_sharding(sharding)
# Todas las operaciones subsecuentes respetan el sharding
```

### 5.4 Chunking Óptimo para el Storage Layer

Las estrategias de chunking en Zarr v3 se eligen según el patrón de acceso:

| Patrón de acceso | Chunking recomendado | Ejemplo |
|---|---|---|
| Batch secuencial | `(batch_size, ...)` | DataLoader: chunks = (64, 224, 224, 3) |
| Slice temporal | `(..., time_chunk)` | Series temporal: chunks = (1000, 128) |
| Acceso aleatorio | Chunks pequeños + sharding | Lookup por ID: chunks = (1,) con shard de 1024 |
| Lectura completa | Un solo chunk | Metadatos: chunks = shape completa |

---

## 6. Interoperabilidad

### 6.1 Bridge con Arrow

```
TensorFrame (campos 1D/escalares) ←→ Arrow RecordBatch
  - Campos escalares: zero-copy vía DLPack o buffer protocol
  - Campos string: conversión de offsets (Arrow usa int32/int64, TF igual)
  - Campos tensor: sin equivalente directo en Arrow; se exportan como
    FixedSizeList o extensión personalizada
```

### 6.2 Bridge con NumPy/Pandas

```
TensorFrame ←→ Pandas DataFrame
  - Solo campos 1D escalares son convertibles directamente
  - Campos tensor se exportan como columnas de object arrays (con warning)
  - frame.to_pandas(flatten=True) aplana campos ND a múltiples columnas
```

### 6.3 Bridge con Frameworks ML

```
TensorFrame → JAX Arrays: frame.to_jax_arrays(features=[...], target=...)
TensorFrame → PyTorch:    frame.to_torch_tensors() (vía DLPack, zero-copy)
TensorFrame → TensorFlow: frame.to_tf_tensors() (vía DLPack)
```

---

## 7. Comparación con Alternativas Existentes

| Aspecto | Pandas | Xarray | TensorFrame (propuesta) |
|---|---|---|---|
| Backend computacional | NumPy (CPU) | NumPy/Dask (CPU) | JAX (CPU/GPU/TPU) |
| Dimensionalidad | 1D (Series), 2D (DataFrame) | N-D | N-D |
| Tipos anidados | Limitado (ExtensionArray) | No | Sí (list, struct, tensor) |
| Evaluación lazy | No | Con Dask (limitado) | I/O lazy, cómputo eager+JIT |
| Diferenciación automática | No | No | Sí (vía JAX grad) |
| Vectorización automática | No | No | Sí (vía JAX vmap) |
| Paralelismo multi-GPU | No | Dask distributed | JAX sharding nativo |
| Storage layer | Archivos planos | NetCDF/Zarr | TensorStore + Zarr v3 |
| IPC zero-copy | Vía Arrow | No estandarizado | TensorBatch (propuesta) |

---

## 8. Riesgos y Mitigaciones

### 8.1 Complejidad del Registro Pytree

**Riesgo:** Mantener la compatibilidad pytree al evolucionar el esquema de TensorFrame es frágil. Los `tree_unflatten` deben reconstruir objetos correctamente incluso cuando JAX pasa objetos placeholder (como `object()`) durante el trazado.

**Mitigación:** Usar `object.__setattr__` en el unflatten (no el constructor), siguiendo el patrón de Equinox. Separar estrictamente datos dinámicos (hojas) de metadatos estáticos (aux). Test exhaustivo con todas las transformaciones JAX.

### 8.2 Shapes Dinámicas y JIT

**Riesgo:** JAX `jit` requiere shapes estáticas. Los TensorFrames con campos de longitud variable (listas) no pueden compilarse directamente.

**Mitigación:** Ofrecer `.to_padded()` que convierte campos variables a formato padded + mask con forma estática. Documentar claramente que las operaciones `jit` requieren formas conocidas y proveer guías de migración.

### 8.3 Overhead de la Capa de Abstracción

**Riesgo:** Las múltiples capas (TensorFrame → LazyExpr → JAX → XLA) agregan latencia en operaciones pequeñas.

**Mitigación:** Para operaciones simples sobre datos ya materializados, proveer un "fast path" que bypasea el grafo de expresiones y ejecuta directamente. Benchmark continuo contra operaciones JAX desnudas para mantener el overhead por debajo del 5%.

### 8.4 Curva de Aprendizaje

**Riesgo:** Los usuarios de Pandas esperan mutabilidad y operaciones in-place.

**Mitigación:** Documentación extensiva con guía de migración "Pandas → TensorFrame". API de conveniencia como `.pipe()` para chaining funcional. Mensajes de error claros cuando se intenta mutación.

---

## 9. Roadmap de Implementación Sugerido

| Fase | Alcance | Dependencias |
|---|---|---|
| **Fase 0: Fundamentos** | NDSchema, NDType, Index, DimSpec. Serialización a/desde JSON. | Ninguna |
| **Fase 1: Core Inmutable** | TensorFrame básico con campos JAX Array. Registro pytree. Indexing básico (sel/isel). with_column, drop. | JAX |
| **Fase 2: Storage L1** | Integración con TensorStore. Open/save desde Zarr v3. Lazy loading. Cache. | TensorStore, zarr-python |
| **Fase 3: Operaciones** | groupby, merge, concat, apply, map. Kernel registry. | Fase 1 |
| **Fase 4: ML Pipeline** | normalize, split, encode_categorical, iter_batches, to_jax_arrays. Prefetch. | Fases 1-3 |
| **Fase 5: Interop** | Arrow bridge, Pandas bridge, DLPack, Parquet I/O, CSV I/O. | PyArrow, Pandas |
| **Fase 6: Avanzado** | LazyExpr graph. Sharding automático. TensorBatch IPC. Optimización de performance. | Fases 1-5 |

---

## 9.1 Profundización de Diseño: Áreas Complementarias

Las siguientes secciones abordan aspectos del diseño que requieren especificación adicional para garantizar robustez en la implementación.

### 9.1.1 Manejo de Errores y Validación en el Storage Layer

El Storage Layer de 3 capas introduce múltiples puntos de falla que deben manejarse de forma predecible.

**Jerarquía de excepciones:**

```
TensorFrameError (base)
├── SchemaError
│   ├── SchemaValidationError      # Esquema inválido al construir
│   ├── SchemaMismatchError        # Esquemas incompatibles en merge/concat
│   └── SchemaEvolutionError       # Migración de esquema fallida
├── StorageError
│   ├── MaterializationError       # Fallo al promover Cold/Warm → Hot
│   ├── ChunkCorruptionError       # Checksum inválido al leer chunk
│   ├── IOTimeoutError             # Timeout en operación de I/O
│   ├── DeviceMemoryError          # Sin memoria en dispositivo para materializar
│   └── PersistenceError           # Fallo al escribir a Cold storage
├── ComputeError
│   ├── ShapeError                 # Formas incompatibles en operación
│   ├── DtypeError                 # Tipos incompatibles
│   └── JITTraceError              # Error durante trazado JIT
└── IndexError
    ├── LabelNotFoundError         # Etiqueta no existe en Index
    └── DimensionError             # Dimensión referenciada no existe
```

**Políticas de recuperación por nivel de storage:**

| Nivel | Tipo de fallo | Política |
|---|---|---|
| Hot → Warm (eviction) | DeviceMemoryError | Retry con eviction LRU forzada; si falla, degradar campo más antiguo |
| Warm → Hot (materialización) | IOTimeoutError | Retry con backoff exponencial (3 intentos, 1s/2s/4s); luego MaterializationError |
| Cold → Warm (carga) | ChunkCorruptionError | Verificar checksum CRC32C; si falla, reportar chunk específico y ofrecer recarga parcial |
| Hot → Cold (persistencia) | PersistenceError | Write-ahead log para operaciones parciales; retry atómico por chunk |

**Validación de integridad:**

Cada chunk en Zarr v3 almacena un checksum CRC32C en sus metadatos. Al leer, TensorStore verifica el checksum antes de decodificar. Si falla:
1. Se marca el chunk como corrupto en un registro interno.
2. Se intenta releer desde el backend de storage (por si fue un error transitorio de red).
3. Si persiste, se lanza `ChunkCorruptionError` con la posición del chunk y su ruta en el store.

### 9.1.2 Concurrencia y Thread-Safety

**Modelo de concurrencia:**

TensorFrame adopta un modelo de **inmutabilidad + message passing** que minimiza la necesidad de locks:

- **Lecturas concurrentes:** Seguras por diseño. Los TensorFrames son inmutables, por lo que múltiples threads pueden leer simultáneamente sin riesgo de data races.
- **Escrituras a Cold storage:** Delegadas a TensorStore, que implementa **optimistic concurrency** con transacciones por chunk. Dos procesos pueden escribir chunks distintos del mismo array simultáneamente.
- **Escrituras concurrentes al mismo chunk:** TensorStore usa compare-and-swap a nivel de storage backend. Si hay conflicto, el escritor más lento recibe un error de concurrencia y debe reintentar.

**Coordinación entre threads Python:**

```
ThreadPool (I/O)          ThreadPool (Cómputo)        Thread Principal
     │                          │                           │
     │  fetch chunks async      │                           │
     │◄─────────────────────────┼───────────────────────────┤ request data
     │                          │                           │
     │  decode + decompress     │                           │
     ├─────────────────────────►│  device_put (DMA)         │
     │                          ├──────────────────────────►│ datos listos
     │                          │                           │ ejecutar kernel
```

- El GIL de Python se libera durante operaciones de I/O (TensorStore es C++) y durante cómputo JAX (XLA ejecuta fuera del GIL).
- El `asyncio` event loop de Zarr-Python 3 se ejecuta en un thread dedicado, evitando bloquear el thread principal.

**Multi-proceso:**

Para acceso desde múltiples procesos Python (e.g., workers de un DataLoader):
- Cada proceso abre su propia instancia de TensorStore con su propio cache pool.
- Las escrituras usan el mecanismo de optimistic concurrency de TensorStore.
- No se comparte estado mutable entre procesos; cada uno tiene su propia copia de los metadatos del TensorFrame.

### 9.1.3 Memory Budget y Backpressure

**Monitoreo de memoria del dispositivo:**

```python
@dataclass(frozen=True)
class MemoryBudget:
    device_limit: int          # Bytes máximos en device memory (auto-detectado o configurable)
    host_cache_limit: int      # Bytes máximos para TensorStore cache pool
    high_watermark: float      # Fracción (e.g., 0.85) para iniciar eviction proactiva
    low_watermark: float       # Fracción (e.g., 0.65) objetivo tras eviction
    prefetch_budget: float     # Fracción reservada para prefetch (e.g., 0.15)
```

**Mecanismo de backpressure:**

Cuando la memoria del dispositivo supera `high_watermark`:

1. **Pausa de prefetch:** Se suspenden las lecturas anticipadas de `iter_batches()`.
2. **Eviction LRU:** Se identifican los campos Hot menos recientemente usados y se degradan a Warm (los datos persisten en el cache de TensorStore).
3. **Materialización bloqueante:** Si un campo necesita materializarse pero no hay espacio, se fuerza eviction de otros campos hasta alcanzar `low_watermark`.
4. **Error terminal:** Si tras evictar todo lo posible no hay espacio suficiente, se lanza `DeviceMemoryError` con un mensaje que indica el tamaño requerido vs. disponible.

**Tracking de uso:**

Cada campo Hot mantiene un timestamp de último acceso. Un `MemoryManager` singleton (por dispositivo) rastrea:
- Bytes totales en uso por campos Hot.
- Orden LRU de campos.
- Bytes reservados para prefetch.

### 9.1.4 Estrategia de Testing

**Niveles de testing:**

| Nivel | Alcance | Herramientas |
|---|---|---|
| **Unitario** | NDType, NDSchema, Index, FieldSpec individuales | pytest, hypothesis (property-based) |
| **Integración** | TensorFrame con operaciones completas, pytree round-trips | pytest, jax.test_util |
| **Property-based** | Invariantes de esquema, idempotencia de serialización, roundtrip pytree | hypothesis con estrategias custom |
| **Transformaciones JAX** | Compatibilidad con jit/grad/vmap/pmap sobre TensorFrames | Tests parametrizados por transformación |
| **Storage** | Round-trip Hot↔Warm↔Cold, integridad de chunks, concurrencia | pytest con fixtures de TensorStore/Zarr en memoria |
| **Fuzzing** | Esquemas aleatorios, datos edge-case (NaN, inf, empty, max-size) | hypothesis |
| **Performance** | Benchmarks de overhead vs JAX directo, throughput de I/O | pytest-benchmark, asv |

**Invariantes a verificar en todo test de TensorFrame:**

1. `tree_unflatten(*tree_flatten(frame))` produce un frame idéntico al original.
2. Toda operación que retorna un TensorFrame produce un frame con esquema válido.
3. Los metadatos (schema, indices, dim_order) sobreviven intactos a jit/vmap/grad.
4. La serialización a JSON de NDSchema es idempotente: `from_json(to_json(schema)) == schema`.

### 9.1.5 Serialización del LazyExpr Graph

El grafo de expresiones lazy puede serializarse para:
- **Checkpointing de pipelines:** Guardar un pipeline de transformaciones sin materializar datos.
- **Debugging:** Inspeccionar qué operaciones se van a ejecutar antes de materializar.
- **Reproducibilidad:** Registrar la secuencia exacta de transformaciones aplicadas.

**Formato de serialización:**

```json
{
  "version": "1.0",
  "nodes": [
    {"id": 0, "op": "field_ref", "field": "precio"},
    {"id": 1, "op": "scalar", "value": 1.16, "dtype": "float64"},
    {"id": 2, "op": "mul", "inputs": [0, 1]},
    {"id": 3, "op": "field_ref", "field": "impuesto"},
    {"id": 4, "op": "add", "inputs": [2, 3]}
  ],
  "output": 4
}
```

**Limitaciones:** Las funciones Python arbitrarias (e.g., lambdas en `.apply()`) no son serializables. En esos casos, el nodo se marca como `"op": "opaque_fn"` y el grafo no es reconstructible desde la serialización.

### 9.1.6 Versionado y Migración de Esquema

Cuando un dataset persistido en Zarr v3 necesita evolución de esquema:

**Operaciones de migración soportadas:**

| Operación | Compatibilidad | Implementación |
|---|---|---|
| Agregar campo con default | Backward compatible | Nuevo array Zarr; lecturas de datos antiguos retornan el default |
| Eliminar campo | Forward compatible | Se marca como deprecated; el array Zarr se retiene pero se ignora |
| Renombrar campo | Metadata-only | Actualización del JSON de esquema |
| Cambiar dtype (widening) | Safe cast | Nuevo array con datos convertidos; validación de pérdida de precisión |
| Cambiar dtype (narrowing) | Requiere confirmación | Validación explícita; falla si hay overflow |
| Agregar dimensión | Breaking | Nuevo dataset; migración explícita |
| Cambiar chunking | Storage-only | Re-chunk con `zarr.copy()` optimizado |

**Metadatos de versión:**

```json
{
  "tensorframe_version": "0.1.0",
  "schema_version": 2,
  "schema_history": [
    {"version": 1, "timestamp": "2026-03-01T00:00:00Z", "changes": "initial"},
    {"version": 2, "timestamp": "2026-03-15T00:00:00Z", "changes": "added field 'precio_iva'"}
  ]
}
```

Al abrir un dataset, TensorFrame compara la `schema_version` del archivo con la esperada. Si difieren, ofrece migración automática para operaciones backward-compatible o lanza `SchemaEvolutionError` con instrucciones para migraciones breaking.

---

## 10. Conclusión

TensorFrame propone una arquitectura que hereda la ergonomía de datos etiquetados de Pandas, la extiende a N dimensiones con soporte para tipos anidados (como Arrow), la acelera con JAX para cómputo en GPU/TPU, y la respalda con un Storage Layer de tres niveles que gestiona inteligentemente la materialización de datos según su temperatura de uso.

La clave del diseño es la alineación profunda con el modelo funcional de JAX (inmutabilidad, pytrees, transformaciones composables) sin sacrificar la experiencia de usuario que hizo exitoso a Pandas (ejes etiquetados, indexing expresivo, operaciones de alto nivel). El Storage Layer no es un bolt-on sino un ciudadano de primera clase del framework, analogando el rol de Arrow para datos columnar pero en el espacio de datos multidimensionales y anidados.
