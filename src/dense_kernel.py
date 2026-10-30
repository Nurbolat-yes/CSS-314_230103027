import os
import time
import json
import numpy as np
from salt import get_student_params

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit
    def naive_kernel(A, B, C, M):
        for i in range(M):
            for j in range(M):
                acc = 0.0
                for k in range(M):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc

    @njit
    def interchanged_kernel(A, B, C, M):
        for i in range(M):
            for j in range(M):
                C[i, j] = 0.0
            for k in range(M):
                a_ik = A[i, k]
                for j in range(M):
                    C[i, j] += a_ik * B[k, j]
else:
    def naive_kernel(A, B, C, M):
        for i in range(M):
            for j in range(M):
                acc = 0.0
                for k in range(M):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc

    def interchanged_kernel(A, B, C, M):
        for i in range(M):
            for j in range(M):
                C[i, j] = 0.0
            for k in range(M):
                a_ik = A[i, k]
                for j in range(M):
                    C[i, j] += a_ik * B[k, j]

def main():
    student_id = "230103027"
    params = get_student_params(student_id)
    N = params["N"]
    M = N // 2 
    
    print(f"[INFO] Student ID: {student_id} | Salted N={N} | Sub-block M={M}")
    
    np.random.seed(params["seed"])
    A = np.random.rand(M, M).astype(np.float64)
    B = np.random.rand(M, M).astype(np.float64)
    
    C_naive = np.zeros((M, M), dtype=np.float64)
    C_interchanged = np.zeros((M, M), dtype=np.float64)

    if HAS_NUMBA:
        print("[INFO] Numba JIT detected. Compiling kernels...")
        naive_kernel(A, B, C_naive, min(M, 16))
        interchanged_kernel(A, B, C_interchanged, min(M, 16))
    else:
        print("[WARNING] Numba not found. Pure Python loops will be slow for M = %d" % M)

    print("[INFO] Benchmarking Algorithm A (Naïve i-j-k)...")
    naive_times = []
    for _ in range(3):
        C_naive.fill(0.0)
        start = time.perf_counter_ns()
        naive_kernel(A, B, C_naive, M)
        end = time.perf_counter_ns()
        naive_times.append(end - start)
    median_naive = np.median(naive_times)

    print("[INFO] Benchmarking Algorithm B (Interchanged i-k-j)...")
    interchanged_times = []
    for _ in range(3):
        C_interchanged.fill(0.0)
        start = time.perf_counter_ns()
        interchanged_kernel(A, B, C_interchanged, M)
        end = time.perf_counter_ns()
        interchanged_times.append(end - start)
    median_interchanged = np.median(interchanged_times)

    speedup = median_naive / median_interchanged if median_interchanged > 0 else 0.0

    results = {
        "student_id": student_id,
        "N": N,
        "M": M,
        "median_naive_ns": median_naive,
        "median_interchanged_ns": median_interchanged,
        "speedup": speedup,
        "numba_accelerated": HAS_NUMBA
    }

    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "loop_interchange.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"[SUCCESS] Loop interchange results written to {json_path}")
    print(f"  - Naive Median Time:       {median_naive / 1e6:8.3f} ms")
    print(f"  - Interchanged Median Time: {median_interchanged / 1e6:8.3f} ms")
    print(f"  - Empirical Speedup (S):   {speedup:8.2f}x")

if __name__ == "__main__":
    main()