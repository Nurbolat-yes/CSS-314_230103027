"""Task 1: Warp divergence microbenchmark.
Usage: python task1_divergence.py [N]   (default N = 2**20)
"""
import sys
import time
import numpy as np
from numba import cuda

ITERS = 1000
THREADS = 256


@cuda.jit
def kernel_uniform(d_arr, N):
    idx = cuda.grid(1)
    if idx < N:
        x = d_arr[idx]
        for _ in range(ITERS):
            x = x * 1.0001 + 0.5          # identical path for every thread
        d_arr[idx] = x


@cuda.jit
def kernel_divergent(d_arr, N):
    idx = cuda.grid(1)
    if idx < N:
        x = d_arr[idx]
        if idx % 2 == 0:                  # adjacent threads diverge
            for _ in range(ITERS):
                x = x * 1.0001 + 0.5
        else:
            for _ in range(ITERS):
                x = (x - 0.5) / 1.0001
        d_arr[idx] = x


@cuda.jit
def kernel_warp_aligned(d_arr, N):
    idx = cuda.grid(1)
    if idx < N:
        x = d_arr[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:              # whole warp takes the same branch
            for _ in range(ITERS):
                x = x * 1.0001 + 0.5
        else:
            for _ in range(ITERS):
                x = (x - 0.5) / 1.0001
        d_arr[idx] = x


def bench(kernel, h_data, blocks, trials=10):
    d_arr = cuda.to_device(h_data)        # transfer excluded from timing
    kernel[blocks, THREADS](d_arr, h_data.size)   # warm-up (also JIT compile)
    cuda.synchronize()
    times = []
    for _ in range(trials):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[blocks, THREADS](d_arr, h_data.size)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000.0 * sum(times) / len(times)   # ms


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 20
    h = np.ones(N, dtype=np.float32)
    blocks = (N + THREADS - 1) // THREADS
    ta = bench(kernel_uniform, h, blocks)
    tb = bench(kernel_divergent, h, blocks)
    tc = bench(kernel_warp_aligned, h, blocks)
    print(f"N = {N}")
    print("| Kernel | Avg time (ms) | Slowdown vs A |")
    print("|---|---|---|")
    print(f"| A: Uniform | {ta:.4f} | 1.00x |")
    print(f"| B: Interleaved divergence | {tb:.4f} | {tb/ta:.2f}x |")
    print(f"| C: Warp-aligned branching | {tc:.4f} | {tc/ta:.2f}x |")


if __name__ == "__main__":
    main()
