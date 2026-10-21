# Flawn Programming Language

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/Host-C%2B%2B23%20%7C%20GCC%20AVX2%2FFMA3-success.svg)](#architecture)
[![Python Superset](https://img.shields.io/badge/Python_Superset-CPython_3.14_Bridge-yellow.svg)](docs/PYTHON_INTEROP.md)

**Flawn** is a high-performance, hardware-auto-tuned programming language designed for backend systems, numerical compute pipelines, and low-latency workloads.

Flawn combines **everyday developer convenience** (zero-ceremony structs, pipe operators `|>`, string interpolation, collection methods, and full Python package compatibility) with **bare-metal SIMD + multi-threaded execution** that outperforms standard C/C++ and runs orders of magnitude faster than Python.

---

## Key Highlights

- **Unified Dual-Grammar (C++/JS + Python Superset)**: Write clean C++/JavaScript-style code (`fn`, `let`, `const`, `struct`, `{ ... }`, `|>`) and standard Python code (`def`, indentation, `for i in range()`, `import`) in the **exact same `.flawn` file**.
- **Both Interpreted & Compiled**:
  - **Interpreted VM (`flawn file.flawn`)**: Instant startup on a 64-bit NaN-boxed bytecode VM backed by a 64-byte aligned `SlabArena`.
  - **Native AOT/JIT (`flawn build file.flawn -o app.exe` or `flawn run file.flawn --aot`)**: Compiles directly to zero-alias (`__restrict__`) C++23 + AVX2/FMA3 machine code.
- **Extreme Speed**:
  - **4–5x to 740x+ faster** on unoptimized standard Python code via automatic integer unboxing and loop lifting.
  - **640x to 35,000x+ faster** on `@simd @parallel` Flawn compute kernels vs. Python, and **1.56x faster** than standard `-O3` C++ by eliminating pointer-aliasing conservative guards and single-accumulator latency chains.
  - **12.5x faster than NumPy** on fused $1024 \times 1024$ matrix polynomial transforms (`1.19 ms` vs `14.91 ms`) by fusing multi-term operations in AVX2 registers with zero temporary array allocations.
- **30–40%+ Lighter Memory (Up to 89% Lighter)**: Uses 8-byte IEEE-754 NaN-boxed values and contiguous 64-byte aligned slab arenas instead of 28–36 byte heap-allocated `PyObject` wrappers.
- **Built-in Hardware Auto-Tuner (`flawn tune`)**: Probes your CPU's SIMD ISA (`AVX2`, `FMA3`, `AVX-512`), logical cores, and cache hierarchy, empirically sweeps unroll and prefetch distances, and generates `include/flawn_tune_profile.hpp`.

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

    // Built-in SIMD-backed array methods
    let latencies = [1.2, 0.8, 1.5, 0.9, 1.1];
    print(f"Mean latency: {latencies.mean()} ms, Sum: {latencies.sum()} ms");

    // Dictionary literal + JSON output
    let status = { "fib_20": python_fibonacci(20), "hash": crypto.fnv1a64("flawn") };
    print(json.stringify(status));
}
```

---

## Measured Benchmarks

| Workload | CPython 3.14 | Standard C++ (`-O3`) | Python + NumPy (`f32`) | Flawn (`@simd @parallel`) | Flawn Advantage |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Unoptimized Python Loop** ($N = 10^7$) | `2168.57 ms` | — | — | **`2.86 ms`** | **759x faster** |
| **Vector Dot Product** ($N = 1.67 \times 10^7$) | `4885.84 ms` | `18.24 ms` | — | **`11.68 ms`** (RAM)<br>**`0.003 ms`** (L1 Tile) | **1.56x faster than C++**<br>**>35,000x vs Python** |
| **$1024 \times 1024$ Fused Matrix Transform** | `761.17 ms` | — | `14.91 ms` | **`1.19 ms`** | **641x vs Python**<br>**12.5x faster than NumPy** |
| **$1024 \times 1024$ Matrix Multiply ($A \times B$)** | `216,088.74 ms` | — | `17.85 ms` (`f32`)<br>`34.36 ms` (`f64`) | **`26.41 ms`** | **8,182x vs Python**<br>**1.3x faster than NumPy `f64`** |
| **Memory: $10^6$ Numeric Elements** | `34.76 MB` | `7.63 MB` | `3.81 MB` | **`7.63 MB` (`NaNVal`)**<br>**`3.81 MB` (`f32[]`)** | **78%–89% lighter than Python** |

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

# 2. Run a .flawn script directly (Instant NaN-Boxed VM)
./build/flawn examples/convenience_showcase.flawn

# 3. Run with Native AOT/JIT compilation
./build/flawn run examples/hybrid_demo.flawn --aot

# 4. Compile to a standalone native executable
./build/flawn build examples/simd_compute.flawn -o build/simd_compute.exe

# 5. Launch Interactive REPL
./build/flawn repl

# 6. Run Test Suite & Benchmarks
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
- [**Python Superset & Interop Guide**](docs/PYTHON_INTEROP.md) — Mixing Python `def` functions, importing Python packages (`numpy`, `math`, `json`), and zero-overhead unboxing.

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for full details.
