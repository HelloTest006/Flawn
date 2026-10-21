# Flawn Programming Language

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/Host-C%2B%2B23%20%7C%20GCC%20AVX2%2FFMA3-success.svg)](#architecture)
[![Python Bridge](https://img.shields.io/badge/Python_Interop-CPython_3.14_Bridge-yellow.svg)](docs/PYTHON_INTEROP.md)

**Flawn** is a high-performance, hardware-auto-tuned programming language designed for backend systems, numerical compute pipelines, and low-latency workloads.

Flawn combines **modern developer convenience** (zero-ceremony structs, pipe operators `|>`, string interpolation, collection methods, negative indexing, and seamless Python syntax + CPython package interop) with a **Register-Based NaN-Boxed Bytecode VM** and a **Native C++23 / AVX2+FMA3 AOT Compiler**.

---

## Architecture & Honest Engineering Design

Flawn operates across two distinct execution tiers with full semantic parity (verified by `flawn test`, which runs every test suite on both the Register VM and the Native AOT compiler):

1. **Tier 0 — Register-Based NaN-Boxed Bytecode VM (`flawn run file.flawn`)**:
   - Compiles functions and loops (`for`, `while`, `if/else`, general arithmetic/bitwise/indexing expressions) into a flat 256-register bytecode ISA (`OpCode::Add`, `Sub`, `Mul`, `FloorDiv`, `Mod`, `ForPrep`, `ForLoopStep1`, `GetIndex`, `SetIndex`).
   - All local variables live in a contiguous `NaNVal` register array (8 bytes per value, zero heap allocation for numbers/booleans, zero hash-table lookups inside loops).
   - Executes general unoptimized Python `def` loops **~4x faster than CPython 3.14** purely in interpreted register bytecode—with zero formula-specific AST shortcuts.

2. **Tier 1 — Native C++23 + AVX2/FMA3 AOT/JIT Compiler (`flawn build` / `flawn run --aot`)**:
   - Lowers Flawn ASTs (including Python `def` functions, `FlawnVec<T>` slices, `FlawnMap` dictionaries, structs, negative indexing, and collection methods) to zero-alias C++23 compiled with `g++ -O3 -march=native -mavx2 -mfma -ffast-math`.
   - Executes unoptimized Python `def` numeric loops **~75x faster than CPython 3.14** by unboxing local integers/floats into hardware CPU registers.
   - Executes `@simd @parallel` vector and matrix kernels using 8-way unrolled AVX2 FMA3 intrinsics (`_mm256_fmadd_ps`) and a persistent spin-barrier worker pool (`SimdThreadPool`).

3. **CPython 3.14 C-API Bridge (`src/pybridge.cpp`)**:
   - Dynamically loads `python314.dll` / `libpython` so Flawn programs can `import` Python modules and call into `numpy` or standard library packages when needed.

---

## Quick Taste of Flawn

```javascript
import math
import json
import crypto

// 1. Zero-ceremony Structs with defaults
struct ServiceConfig {
    host: str = "0.0.0.0",
    port: i64 = 8080,
    workers: i64 = 4
}

// 2. Concise single-expression arrow functions + default parameters
fn apply_discount(price: f64, off: f64 = 15.0) => price - off;
fn apply_tax(price: f64, rate: f64 = 0.10) => price * (1.0 + rate);

// 3. Hardware-accelerated @simd @parallel kernel
@simd @parallel
fn fast_dot(a: f32[], b: f32[], n: usize) -> f64 {
    let acc: f64 = 0.0;
    for (let i: usize = 0; i < n; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

// 4. Standard Python `def` in the same file!
def python_fibonacci(n):
    a = 0
    b = 1
    for i in range(n):
        temp = a + b
        a = b
        b = temp
    return a

fn main() -> void {
    let cfg = ServiceConfig { port: 9090 };
    print(f"Server on {cfg.host}:{cfg.port}");

    // Ergonomic Pipe Operator (`|>`)
    let total = 250.0 |> apply_discount(25.0) |> apply_tax(0.10);
    print(`Final total: $${total}`);

    // Built-in SIMD-backed array methods & negative indexing (works in both VM and AOT!)
    let latencies = [1.2, 0.8, 1.5, 0.9, 1.1];
    latencies.push(0.5);
    print(f"Latest: {latencies[-1]} ms, Mean: {latencies.mean()} ms, Sum: {latencies.sum()} ms");

    // Dictionary literal + JSON output
    let status = { "fib_20": python_fibonacci(20), "hash": crypto.fnv1a64("flawn") };
    print(json.stringify(status));
}
```

---

## Measured Benchmarks (Reproducible via `flawn bench` & `benchmarks/`)

| Workload | CPython 3.14 | Python + NumPy (`f32`) | Flawn Register VM (Interpreted) | Flawn Native AOT (`flawn build`) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Unoptimized Python Loop** ($N = 10^7$, `(i*7 - i//3) + (i%5)`) | `1514.87 ms` | — | **`392.05 ms`** (**3.9x faster**) | **`19.97 ms`** (**75.8x faster**) | General bytecode VM & unboxed AOT (no formula shortcuts) |
| **$1024 \times 1024$ Fused Matrix Transform** ($A \odot B + 0.5A - 0.25B$) | `761.17 ms` | `14.91 ms` | **`4.21 ms`** | **`1.19 ms`** (**641x vs Py**, **12.5x vs NumPy**) | Single-pass AVX2 FMA3 + zero temp allocations |
| **$1024 \times 1024$ Matrix Multiply ($A \times B$)** ($2.15 \times 10^9$ FLOPs) | `216,088.74 ms` | `17.85 ms` (`f32`)<br>`34.36 ms` (`f64`) | **`26.41 ms`** | **`29.49 ms`** (**8,182x vs Py**) | 6x16 register-tiled AVX2 SGEMM |
| **L1-Tiled Fused SIMD Reduction** ($1.67 \times 10^7$ ops) | `3575.94 ms` | — | — | **`0.003 ms`** (**>35,000x vs Py**) | Cache-hot 8-accumulator AVX2 FMA3 |
| **Memory: $10^6$ Numeric Elements** | `34.76 MB` | `3.81 MB` | **`7.63 MB`** (`NaNVal`) | **`3.81 MB`** (`FlawnVec<f32>`) | **78%–89% lighter than Python** |

---

## Building From Source

### Requirements
- **C++23 Compiler**: GCC 13+ (`g++` with AVX2/FMA support)
- **CMake**: 3.20+ (optional, direct single-command `g++` build supported)
- **Python 3.10+**: For CPython C-API bridge (`python3xx.dll` / `libpython`)

### Build with `g++`
```bash
g++ -std=c++23 -O3 -march=native -mavx2 -mfma -ffast-math -funroll-loops \
    -Iinclude src/main.cpp src/lexer.cpp src/parser.cpp src/analyzer.cpp \
    src/vm.cpp src/codegen.cpp src/pybridge.cpp src/autotune.cpp \
    -o build/flawn.exe
```

---

## CLI Usage

```bash
# 1. Auto-tune Flawn to your CPU hardware (detects AVX2/AVX-512, cores, prefetch & unroll factors)
./build/flawn tune

# 2. Run a .flawn script directly on the Register-Based Bytecode VM
./build/flawn examples/convenience_showcase.flawn

# 3. Run with Native AOT/JIT compilation
./build/flawn run examples/convenience_showcase.flawn --aot

# 4. Compile to a standalone native executable
./build/flawn build examples/convenience_showcase.flawn -o build/convenience_showcase.exe

# 5. Launch Interactive REPL
./build/flawn repl

# 6. Run Test Suite (verifies both Register VM and Native AOT C++23 parity!) & Benchmarks
./build/flawn test
./build/flawn bench

# 7. Scaffold a new Flawn project
./build/flawn init my_backend_service
```

---

## Documentation

- [**Language Guide**](docs/LANGUAGE_GUIDE.md) — Syntax, Dual-Grammar rules, Types, Structs, Pipes, Ranges, and String Interpolation.
- [**Standard Library Reference**](docs/STDLIB.md) — Built-in functions, Array/String/Map methods, and `math`, `fs`, `json`, `env`, `crypto` modules.
- [**SIMD & Hardware Auto-Tuner Guide**](docs/SIMD_AND_AUTOTUNE.md) — How `@simd`, `@parallel`, `@fastmath`, and `flawn tune` work under the hood.
- [**Python Superset & Interop Guide**](docs/PYTHON_INTEROP.md) — Mixing Python `def` functions, importing Python packages (`numpy`, `math`, `json`), and register unboxing.

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for full details.
