# CS482: Performance Evaluation & Optimization - Lab Practicum

**Student ID:** 230103027  
**Name:** Kemelov Nurbolat  

## Bottleneck Findings Summary
This laboratory practicum evaluated hardware-level performance limits across memory hierarchies, multi-core contention, and scaling models. Key findings include:
1. **Memory Stride:** Transitioning from stride-1 to stride-8+ exposed a severe cache-line cliff, as non-contiguous access patterns bypassed hardware prefetchers and saturated DRAM latency.
2. **Loop Interchange:** Reordering dense matrix multiplication loops from $i-j-k$ to $i-k-j$ drastically improved spatial locality and yielded significant speedups.
3. **False Sharing:** Unpadded multi-threaded concurrent updates on shared cache lines induced severe MESI protocol thrashing, which was successfully remediated via 64-byte padding and thread-local privatization.
4. **Scaling & Roofline:** Strong and weak scaling studies validated Amdahl’s and Gustafson’s laws, while Roofline analysis classified the stencil kernel as memory-bandwidth bound.