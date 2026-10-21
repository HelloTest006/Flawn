# Flawn Standard Library Reference

Flawn includes zero-overhead built-in functions, method intrinsics on Arrays, Strings, and Maps, and standard modules (`math`, `fs`, `json`, `env`, `crypto`) plus full access to Python's ecosystem.

---

## 1. Global Built-in Functions

| Function | Signature | Description |
| :--- | :--- | :--- |
| `print(...)` / `println(...)` | `print(a, b, ...)` | Prints values separated by space followed by newline |
| `len(x)` | `len(arr_or_str_or_map) -> i64` | Returns element count or byte length |
| `str(x)` | `str(any) -> str` | Converts any value, array, map, or struct to string |
| `int(x)` | `int(any) -> i64` | Parses or converts value to 64-bit integer |
| `float(x)` | `float(any) -> f64` | Parses or converts value to 64-bit float |
| `abs(x)` | `abs(num) -> num` | Absolute value |
| `min(a, b)` / `max(a, b)` | `(f64, f64) -> f64` | Minimum or maximum of two numbers |
| `clamp(x, lo, hi)` | `(f64, f64, f64) -> f64` | Clamps `x` to `[lo, hi]` |
| `type_of(x)` | `type_of(any) -> str` | Returns runtime type name (`"i64"`, `"f64"`, `"str"`, `"array"`, `"map"`, or struct name) |
| `clock_ms()` | `clock_ms() -> f64` | High-resolution monotonic timestamp in milliseconds |
| `arena_kb()` | `arena_kb() -> f64` | Current memory allocated inside the 64B-aligned SlabArena in KB |

---

## 2. Array & SIMD Slice Methods

| Method | Example | Description |
| :--- | :--- | :--- |
| `.len()` / `.size()` | `arr.len()` | Number of elements |
| `.sum()` | `arr.sum()` | SIMD-accelerated sum of elements |
| `.mean()` / `.avg()` | `arr.mean()` | Arithmetic mean of elements |
| `.min()` / `.max()` | `arr.min()` | Minimum or maximum element |
| `.dot(other)` | `vec_a.dot(vec_b)` | Multi-threaded AVX2 FMA3 dot product |
| `.push(val)` / `.append(val)` | `arr.push(42.0)` | Appends element in-place inside the SlabArena |

---

## 3. String Methods

| Method | Example | Description |
| :--- | :--- | :--- |
| `.len()` | `s.len()` | Length in bytes |
| `.upper()` | `s.upper()` | Returns uppercase string |
| `.lower()` | `s.lower()` | Returns lowercase string |
| `.trim()` / `.strip()` | `s.trim()` | Strips leading and trailing whitespace |
| `.contains(sub)` | `s.contains("flawn")` | Returns `true` if substring exists |
| `.starts_with(prefix)` | `s.starts_with("http")` | Returns `true` if string starts with `prefix` |
| `.replace(old, new)` | `s.replace("a", "b")` | Replaces all occurrences of `old` with `new` |

---

## 4. Map / Dictionary Methods

| Method | Example | Description |
| :--- | :--- | :--- |
| `.has(key)` / `.contains(key)` | `m.has("id")` | Checks if key exists |
| `.get(key, default)` | `m.get("port", 8080)` | Returns value for `key` or `default` |

---

## 5. Built-in Modules

### `import math`
- Constants: `math.pi`, `math.e`
- Functions: `math.sqrt(x)`, `math.sin(x)`, `math.cos(x)`, `math.tan(x)`, `math.exp(x)`, `math.log(x)`, `math.floor(x)`, `math.ceil(x)`, `math.round(x)`, `math.hypot(x, y)`, `math.pow(x, y)`

### `import fs`
- `fs.read(path) -> str`: Reads entire file into an arena string.
- `fs.write(path, content) -> bool`: Writes string content to disk.
- `fs.exists(path) -> bool`: Checks if file or directory exists.

### `import json`
- `json.stringify(obj) -> str` / `json.dumps(obj) -> str`: Serializes Flawn maps, structs, arrays, and primitives to JSON.

### `import env`
- `env.get(var_name, default_val = "") -> str`: Reads OS environment variable.

### `import crypto`
- `crypto.fnv1a64(input_str) -> i64`: Fast 64-bit FNV-1a non-cryptographic hash for backend routing and sharding.
