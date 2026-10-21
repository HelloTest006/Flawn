# Python Syntax & Ecosystem Interoperability

Flawn features a Unified Dual-Grammar parser that accepts standard Python `def` functions, `for i in range(...)` loops, `while` loops, `if/elif/else` blocks, and `import` statements alongside C++/JS-style Flawn code, and links against CPython's C-API (`python314.dll` / `libpython`).

---

## 1. How Flawn Executes Unoptimized Python Code Faster

When Flawn parses a standard Python `def` function:
```python
def py_compute(n):
    total = 0
    for i in range(n):
        total += (i * 7 - (i // 3)) + (i % 5)
    return total
```
Flawn executes it in one of two ways depending on mode—without any hardcoded formula shortcuts:

1. **Interpreted Mode (`flawn run file.flawn` — ~4x Faster than CPython)**:
   - `FunctionCompiler` (`src/vm.cpp`) compiles the Python AST into Flawn's flat 256-register bytecode ISA (`OpCode::Mul`, `FloorDiv`, `Sub`, `Mod`, `Add`, `ForLoopStep1`).
   - All local variables (`n`, `total`, `i`) are mapped to contiguous 8-byte IEEE-754 NaN-boxed registers (`R[0]`, `R[1]`, `R[2]`), eliminating Python's 28-byte heap `PyLongObject` allocations, reference-counting overhead, and `range` iterator object boxing.
2. **Compiled Mode (`flawn build` or `flawn run --aot` — ~75x Faster than CPython)**:
   - `Analyzer` (`src/analyzer.cpp`) infers 64-bit integer/float types for local variables and lowers the function directly to native C++23 (`int64_t` hardware registers), allowing GCC `-O3` to vectorize and unroll the loop into native machine instructions.

---

## 2. Importing Standard Python & Pip Packages

Use standard `import <package>` syntax to load any module installed in your Python environment (`numpy`, `statistics`, etc.):

```javascript
import math

fn main() -> void {
    let h = math.hypot(3.0, 4.0);
    print(f"Result: {h}");
}
```

For inline multi-line Python scripts or NumPy expressions that share state with Flawn, use `py_exec(code)` and `py_eval_f64(expr)`:

```javascript
py_exec("import numpy as np");
py_exec("arr = np.arange(1000, dtype=np.float64)");
let total = py_eval_f64("float(np.sum(arr))");
print(f"NumPy sum = {total}");
```
