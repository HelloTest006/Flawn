# Flawn Language Guide

Flawn is designed around two core principles:
1. **Extreme Developer Convenience**: Write expressive, low-boilerplate code with modern ergonomics (`|>`, `..`, `f"..."`, `=>`, collection methods, and Python compatibility).
2. **Extreme Hardware Performance**: Zero-alias arena slices, multi-accumulator AVX2/FMA3 vectorization, and persistent spin-barrier worker threads.

---

## 1. Unified Dual-Grammar (C++/JS + Python in One File)

Flawn's lexer and parser track brace depth (`{ ... }`) alongside Python indentation (`INDENT` / `DEDENT`):
- Inside `{ ... }` blocks or `fn` / `let` / `const` declarations, newlines are treated as whitespace (C++/JavaScript rules), and semicolons `;` are optional.
- In colon-terminated blocks (`def`, `if cond:`, `for i in range(n):`), standard Python indentation rules apply.

```javascript
// Flawn C++/JS style function
fn calculate_area(w: f64, h: f64 = 10.0) -> f64 {
    return w * h;
}

// Concise single-expression arrow function
fn cube(x: f64) => x * x * x;

# Standard Python function in the exact same file
def py_sum_evens(limit):
    total = 0
    for i in range(0, limit, 2):
        total += i
    return total
```

---

## 2. Variables & Type Inference

Use `let` (mutable), `const` (immutable), or Python-style implicit assignment. Types are inferred automatically if omitted, or can be explicitly annotated:

```javascript
let count = 42;                 // Inferred as i64 (unboxed 64-bit integer)
const pi_approx: f64 = 3.14159; // Explicit f64
let name = "Flawn";             // Arena-allocated string
```

### Supported Primitive & Slice Types
| Flawn Type | C++23 Lowered Type | Description |
| :--- | :--- | :--- |
| `i64` / `int` | `int64_t` | 64-bit signed integer |
| `i32` | `int32_t` | 32-bit signed integer |
| `usize` | `size_t` | Unsigned pointer-sized index |
| `f64` / `float` | `double` | 64-bit IEEE-754 float |
| `f32` | `float` | 32-bit IEEE-754 float |
| `bool` | `bool` | Boolean (`true`/`false` or `True`/`False`) |
| `str` / `string` | `std::string` / `VmString` | UTF-8 string |
| `f32[]` | `float* FLAWN_RESTRICT` | 64-byte aligned SIMD float32 slice |
| `f64[]` | `double* FLAWN_RESTRICT` | 64-byte aligned SIMD float64 slice |
| `i64[]` | `int64_t* FLAWN_RESTRICT` | 64-byte aligned SIMD int64 slice |

---

## 3. String Interpolation (F-Strings & Template Literals)

Flawn supports both Python `f"...{expr}..."` and JavaScript `` `...${expr}...` `` template strings with full arbitrary expressions inside `{}`:

```javascript
let reqs = 125000;
let latency = 0.42;
print(f"Throughput: {reqs * 2} req/s at {latency} ms");
print(`Status: ${latency < 1.0 ? "OPTIMAL" : "DEGRADED"}`);
```

---

## 4. Pipe Operator (`|>`)

Chain data transformations left-to-right without deeply nested parentheses. `x |> f(a, b)` automatically passes `x` as the first argument `f(x, a, b)`:

```javascript
fn add_base(x: f64, base: f64) => x + base;
fn scale(x: f64, factor: f64) => x * factor;

let result = 10.0 |> add_base(5.0) |> scale(3.0); // 45.0
```

---

## 5. Ranges (`..` and `..=`) & Loops

Flawn supports 4 loop styles:
1. **Exclusive Range (`0..n`)**:
   ```javascript
   for i in 0..10 {
       print(i); // 0 to 9
   }
   ```
2. **Inclusive Range (`1..=n`)**:
   ```javascript
   for i in 1..=10 {
       print(i); // 1 to 10
   }
   ```
3. **For-Each Collection Loop (`for item in arr`)**:
   ```javascript
   let items = [10, 20, 30];
   for x in items {
       print(x);
   }
   ```
4. **C++/JS `for (let i = 0; i < n; i++)` and Python `for i in range(n):`** (both support `break` and `continue`).

---

## 6. Structs, Arrays, and Maps

### Structs with Default Field Values
```javascript
struct Endpoint {
    host: str = "127.0.0.1",
    port: i64 = 8080,
    tls: bool = true
}

let ep = Endpoint { port: 443 };
print(ep.host, ep.port);
```

### Dynamic Arrays & Negative Indexing
```javascript
let scores = [10.0, 20.0, 30.0, 40.0];
scores.push(50.0);
print(scores[-1]);     // 50.0 (last element)
print(scores.sum());   // 150.0
print(scores.mean());  // 30.0
```

### Dictionaries / Maps
```javascript
let config = {
    "env": "production",
    "max_conns": 4096
};
config["max_conns"] = 8192;
if config.has("env") {
    print(config["env"]);
}
```

---

## 7. Ternary & Null-Coalescing Operators

- **C++/JS Ternary**: `cond ? a : b`
- **Python Ternary**: `a if cond else b`
- **Null-Coalescing**: `val ?? default_val`
- **Built-in Assertions**: `assert condition, "optional error message"`
