import os
import time
import numpy as np
from salt import get_student_params

def main():
    student_id = "230103027"
    params = get_student_params(student_id)
    buffer_elements = params["buffer_elements"]
    
    print(f"[INFO] Allocating flat array of {buffer_elements:,} float64 elements (~134 MB)...")
    data = np.ones(buffer_elements, dtype=np.float64)
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    
    results = []
    
    print("[INFO] Running stride latency cliff benchmark...")
    for s in strides:
        
        slice_view = data[::s]
        total_elements_touched = len(slice_view)
        
        for _ in range(5):
            _ = slice_view.sum()
            
        times_ns = []
        for _ in range(5):
            start = time.perf_counter_ns()
            _ = slice_view.sum()
            end = time.perf_counter_ns()
            times_ns.append(end - start)
            
        median_time_ns = np.median(times_ns)
        latency_per_element_ns = median_time_ns / total_elements_touched
        
        results.append((s, median_time_ns, total_elements_touched, latency_per_element_ns))
        print(f"Stride {s:3d} | Median Time: {median_time_ns/1e6:7.3f} ms | Latency/Elem: {latency_per_element_ns:6.3f} ns")

    os.makedirs("results", exist_ok=True)
    tsv_path = os.path.join("results", "stride_results.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("stride\tmedian_time_ns\ttouched_elements\tlatency_per_element_ns\n")
        for s, t, el, lat in results:
            f.write(f"{s}\t{t}\t{el}\t{lat:.4f}\n")
            
    print(f"[SUCCESS] Stride results successfully written to {tsv_path}")

if __name__ == "__main__":
    main()