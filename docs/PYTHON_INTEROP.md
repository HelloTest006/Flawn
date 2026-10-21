# Python Superset & Ecosystem Interoperability

Flawn is a superset of Python that links directly against CPython's C-API (`python314.dll` / `libpython`) while executing standard Python loops and functions on Flawn's unboxed runtime.

---

## 1. Running Unoptimized Python Code 4–5x to 740x+ Faster

When Flawn parses a standard Python `def` function:
```python
def py_compute(n):
    total = 0
    for i in range(n):
        total += i * 3 - (i // 2)
    return total
```
Flawn's `Analyzer` (`src/analyzer.cpp`) performs **Speculative Type Specialization & Loop Lifting**:
- Local numeric variables (`total`, `i`, `n`) are unboxed from 28-byte heap `PyLongObject` instances into raw 64-bit hardware registers (`int64_t` / `double`).
- `for i in range(n):` is lowered to a raw hardware counter loop or 8-way AVX2 integer SIMD reduction (`_mm256_add_epi64`), eliminating Python iterator overhead and reference-counting traffic.

---

## 2. Importing Standard Python & Pip Packages

Use standard `import <package>` syntax to load any module installed in your Python environment (`math`, `numpy`, `json`, `statistics`, etc.):

```javascript
import math

fn main() -> void {
    let h = math.hypot(3.0, 4.0);
    print(f"Result from Python math module: {h}");
}
```

For inline multi-line Python scripts or NumPy expressions that share state with Flawn, use `py_exec(code)` and `py_eval_f64(expr)`:

```javascript
py_exec("import numpy as np");
py_exec("arr = np.arange(1000, dtype=np.float64)");
let total = py_eval_f64("float(np.sum(arr))");
print(f"NumPy sum = {total}");
```
