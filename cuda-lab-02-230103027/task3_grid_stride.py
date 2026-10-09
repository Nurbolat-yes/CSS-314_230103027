"""Task 3: Grid-stride loop scaling of an arbitrary-size vector."""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64   # 64 * 256 = 16,384 threads total (fixed)


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        d_arr, np.float32(factor), h_arr.size)
    cuda.synchronize()
    return d_arr.copy_to_host()


def main():
    N = 1_000_000          # far more elements than the 16,384 launched threads
    factor = 4.25
    res = run_grid_stride(np.ones(N, dtype=np.float32), factor)
    assert np.allclose(res, factor), "not all elements equal factor"
    print(f"TASK 3 PASSED: all {N} elements == {factor} "
          f"(threads launched: {BLOCKS_PER_GRID * THREADS_PER_BLOCK})")


if __name__ == "__main__":
    main()
