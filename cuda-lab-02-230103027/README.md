# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103027
**Allocated GPU Node:** <e.g., Tesla T4>  <-- fill in (run `!nvidia-smi` in Colab)
**CUDA Compute Capability:** <e.g., 7.5>  <-- fill in
**Official Verification Token:** <PASTE_TOKEN_HERE>  <-- from `python verify_submission.py` on your GPU

## Task 1: Warp Divergence Benchmark
N = 1,048,576 float32 elements, 1000 iterations per element, 256 threads/block, 10-trial average after warm-up (kernel-only time).

| Kernel | Avg time (ms) | Slowdown vs A |
|---|---|---|
| A: Uniform | <fill> | 1.00x |
| B: Interleaved divergence (idx % 2) | <fill> | <fill> |
| C: Warp-aligned branching (warp_id % 2) | <fill> | <fill> |

**Analysis:** In Kernel B, even and odd threads in the same warp take different branches, so the hardware executes both paths one after another with inactive threads masked off; the warp pays the cost of both paths. In Kernel C the branch condition is constant within each 32-thread warp, so every warp runs only one path and there is no intra-warp serialization. Therefore C should run close to A, and B should be the slowest.

## Task 2: 1D Stencil
3-point filter `0.25*left + 0.5*center + 0.25*right` with edge clamping (halo replication) at idx = 0 and idx = N-1, N = 10007.
Result: `TASK 2 PASSED: MAX DELTA = <fill from your run>`

## Task 3: Grid-Stride Scaling
N = 1,000,000 elements scaled by 4.25 using only 64 x 256 = 16,384 threads. Each thread starts at `cuda.grid(1)` and advances by `cuda.gridsize(1)`, so any N is covered with a fixed launch size. All elements verified equal to the factor.

## Task 4: 2D Sobel-X
1024 x 1024 float32 image, 16x16 blocks, grid = ceil(cols/16) x ceil(rows/16). Interior pixels use the Sobel-X kernel [[-1,0,1],[-2,0,2],[-1,0,1]]; border pixels are set to 0. Verified against a NumPy reference.
