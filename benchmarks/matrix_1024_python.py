import time
import numpy as np

N = 1024
SCALE = 1.0 / 256.0  # 0.00390625 (exact binary float)

print("========================================================================")
print(f"  PYTHON 3.14 BENCHMARK: {N}x{N} MATRIX PROCESSING (WITHOUT & WITH NUMPY)")
print("========================================================================")

# 1. Initialize 1024x1024 Matrices A and B in Pure Python (list of lists)
t_init0 = time.perf_counter()
A_py = [[((i * 7 + j * 13) & 255) * SCALE for j in range(N)] for i in range(N)]
B_py = [[((i * 11 + j * 5) & 255) * SCALE for j in range(N)] for i in range(N)]
C_py = [[0.0] * N for _ in range(N)]
t_init1 = time.perf_counter()
print(f"[Init] Pure Python 1024x1024 Lists Created in {(t_init1 - t_init0)*1000.0:.4f} ms\n")

# ------------------------------------------------------------------------
# WORKLOAD A: 1024x1024 Fused Elementwise Matrix Transform & Reduction
# C[i][j] = (A[i][j] * B[i][j] + 0.5 * A[i][j]) - 0.25 * B[i][j]
# ------------------------------------------------------------------------
print("--- Workload A: 1024x1024 Fused Matrix Transform & Sum ---")

# A1. Pure Python (WITHOUT NumPy)
t0 = time.perf_counter()
py_elem_sum = 0.0
for i in range(N):
    row_a = A_py[i]
    row_b = B_py[i]
    row_c = C_py[i]
    for j in range(N):
        v = (row_a[j] * row_b[j] + 0.5 * row_a[j]) - 0.25 * row_b[j]
        row_c[j] = v
        py_elem_sum += v
t1 = time.perf_counter()
py_no_numpy_elem_ms = (t1 - t0) * 1000.0

# A2. Python WITH NumPy (float32 & float64)
A_np32 = np.array(A_py, dtype=np.float32)
B_np32 = np.array(B_py, dtype=np.float32)
A_np64 = np.array(A_py, dtype=np.float64)
B_np64 = np.array(B_py, dtype=np.float64)

# Warmup
_ = float(np.sum((A_np32 * B_np32 + 0.5 * A_np32) - 0.25 * B_np32, dtype=np.float64))

iters = 20
t0 = time.perf_counter()
for _ in range(iters):
    C_np32 = (A_np32 * B_np32 + np.float32(0.5) * A_np32) - np.float32(0.25) * B_np32
    np32_elem_sum = float(np.sum(C_np32, dtype=np.float64))
t1 = time.perf_counter()
py_numpy32_elem_ms = ((t1 - t0) * 1000.0) / iters

t0 = time.perf_counter()
for _ in range(iters):
    C_np64 = (A_np64 * B_np64 + 0.5 * A_np64) - 0.25 * B_np64
    np64_elem_sum = float(np.sum(C_np64, dtype=np.float64))
t1 = time.perf_counter()
py_numpy64_elem_ms = ((t1 - t0) * 1000.0) / iters

print(f"  [Python WITHOUT NumPy] Time: {py_no_numpy_elem_ms:10.4f} ms | Checksum: {py_elem_sum:.4f}")
print(f"  [Python WITH NumPy 32] Time: {py_numpy32_elem_ms:10.4f} ms | Checksum: {np32_elem_sum:.4f}")
print(f"  [Python WITH NumPy 64] Time: {py_numpy64_elem_ms:10.4f} ms | Checksum: {np64_elem_sum:.4f}\n")

# ------------------------------------------------------------------------
# WORKLOAD B: 1024x1024 Matrix Multiplication (C = A @ B, 2.15 Billion FLOPs)
# ------------------------------------------------------------------------
print("--- Workload B: 1024x1024 Matrix Multiplication (C = A @ B) ---")

# B1. Pure Python (WITHOUT NumPy)
# Full 1024^3 = 1,073,741,824 iterations in pure Python takes ~95 seconds.
# We time 32 full rows of 1024x1024 (32 x 1024 x 1024 = 33.55M multiply-adds) and scale by 32x,
# and also compute the exact reference checksum.
SUB_ROWS = 32
# Transpose B for cache-friendly pure Python row dot-product
B_T_py = [[B_py[k][j] for k in range(N)] for j in range(N)]

t0 = time.perf_counter()
partial_py_mm_sum = 0.0
for i in range(SUB_ROWS):
    row_a = A_py[i]
    for j in range(N):
        col_b = B_T_py[j]
        acc = 0.0
        for k in range(N):
            acc += row_a[k] * col_b[k]
        partial_py_mm_sum += acc
t1 = time.perf_counter()
py_no_numpy_matmul_ms = (t1 - t0) * 1000.0 * (N / SUB_ROWS)

# B2. Python WITH NumPy (`A @ B` BLAS SGEMM / DGEMM)
_ = A_np32 @ B_np32
t0 = time.perf_counter()
for _ in range(10):
    C_mm32 = A_np32 @ B_np32
t1 = time.perf_counter()
py_numpy32_matmul_ms = ((t1 - t0) * 1000.0) / 10.0
np32_mm_sum = float(np.sum(C_mm32, dtype=np.float64))

_ = A_np64 @ B_np64
t0 = time.perf_counter()
for _ in range(10):
    C_mm64 = A_np64 @ B_np64
t1 = time.perf_counter()
py_numpy64_matmul_ms = ((t1 - t0) * 1000.0) / 10.0
np64_mm_sum = float(np.sum(C_mm64, dtype=np.float64))

print(f"  [Python WITHOUT NumPy] Time: {py_no_numpy_matmul_ms:10.4f} ms | (32-row exact sum={partial_py_mm_sum:.4f}, full sum={np64_mm_sum:.4f})")
print(f"  [Python WITH NumPy 32] Time: {py_numpy32_matmul_ms:10.4f} ms | Checksum: {np32_mm_sum:.4f}")
print(f"  [Python WITH NumPy 64] Time: {py_numpy64_matmul_ms:10.4f} ms | Checksum: {np64_mm_sum:.4f}")
print("========================================================================")
