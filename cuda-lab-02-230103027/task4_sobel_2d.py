"""Task 4: 2D Sobel-X filter."""
import numpy as np
from numba import cuda


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if row > 0 and row < rows - 1 and col > 0 and col < cols - 1:
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1] + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1] + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1] + 1.0 * d_in[row + 1, col + 1]
            )
        else:
            d_out[row, col] = 0.0


def run_sobel(h_img):
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)
    threads_2d = (16, 16)
    blocks_2d = ((cols + threads_2d[0] - 1) // threads_2d[0],
                 (rows + threads_2d[1] - 1) // threads_2d[1])
    sobel_x_kernel[blocks_2d, threads_2d](d_in, d_out, rows, cols)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_sobel(img):
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (
        -img[:-2, :-2] + img[:-2, 2:]
        - 2 * img[1:-1, :-2] + 2 * img[1:-1, 2:]
        - img[2:, :-2] + img[2:, 2:]
    )
    return out


def main():
    rows, cols = 1024, 1024
    rng = np.random.default_rng(0)
    img = rng.random((rows, cols), dtype=np.float32)
    gpu = run_sobel(img)
    ref = cpu_sobel(img)
    assert np.allclose(gpu, ref, atol=1e-4)
    print(f"TASK 4 PASSED: shape {gpu.shape}, MAX DELTA = {np.max(np.abs(gpu - ref))}")


if __name__ == "__main__":
    main()
