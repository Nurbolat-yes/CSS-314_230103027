import os
import time
import json
import math
import multiprocessing as mp
import numpy as np
from salt import get_student_params

def stencil_worker(chunk_data):
    matrix = chunk_data
    for _ in range(3):
        matrix = 0.25 * (np.roll(matrix, 1, axis=0) + np.roll(matrix, -1, axis=0) +
                         np.roll(matrix, 1, axis=1) + np.roll(matrix, -1, axis=1))
    return matrix

def run_strong_scaling(N, P_list):
    print(f"[INFO] Running Strong Scaling for N={N} with P in {P_list}...")
    np.seed = 42
    full_matrix = np.random.rand(N, N).astype(np.float64)
    
    results = {}
    T1 = 0.0

    for P in P_list:
        chunks = np.array_split(full_matrix, P, axis=0)
        
        start = time.perf_counter()
        if P == 1:
            _ = [stencil_worker(c) for c in chunks]
        else:
            with mp.Pool(processes=P) as pool:
                _ = pool.map(stencil_worker, chunks)
        end = time.perf_counter()
        
        TP = end - start
        if P == 1:
            T1 = TP
        results[P] = TP
        
    return T1, results

def run_weak_scaling(N_base, P_list):
    print(f"[INFO] Running Weak Scaling with base N={N_base} for P in {P_list}...")
    results = {}
    T1_weak = 0.0

    for idx, P in enumerate(P_list):
        # NP = ceil(N_base * sqrt(P))
        NP = int(math.ceil(N_base * math.sqrt(P)))
        full_matrix = np.random.rand(NP, NP).astype(np.float64)
        chunks = np.array_split(full_matrix, P, axis=0)
        
        start = time.perf_counter()
        if P == 1:
            _ = [stencil_worker(c) for c in chunks]
        else:
            with mp.Pool(processes=P) as pool:
                _ = pool.map(stencil_worker, chunks)
        end = time.perf_counter()
        
        TP = end - start
        if P == 1:
            T1_weak = TP
        results[P] = (NP, TP)
        
    return T1_weak, results

def main():
    student_id = "230103027"
    params = get_student_params(student_id)
    N = params["N"]
    
    max_cores = psutil_cores = os.cpu_count() or 4
    P_list = [p for p in [1, 2, 4, 8, 16] if p <= max_cores]
    if 1 not in P_list:
        P_list.insert(0, 1)

    T1, strong_data = run_strong_scaling(N, P_list)
    
    strong_rows = ["P\truntime_s\tspeedup_SP\tefficiency_EP\tamdahl_serial_fraction_s"]
    strong_metrics = []
    
    for P in P_list:
        TP = strong_data[P]
        SP = T1 / TP if TP > 0 else 0.0
        EP = (SP / P) * 100.0
        
        if P == 1:
            s = 0.0
        else:
            s = ((1.0 / SP) - (1.0 / P)) / (1.0 - (1.0 / P))
            s = max(0.0, min(1.0, s)) # ограничим пределы [0, 1]
            
        strong_metrics.append((P, TP, SP, EP, s))
        strong_rows.append(f"{P}\t{TP:.6f}\t{SP:.4f}\t{EP:.2f}%\t{s:.4f}")

    T1_weak, weak_data = run_weak_scaling(N, P_list)
    
    weak_rows = ["P\tmatrix_size_NP\truntime_weak_s\tscaled_speedup_SP"]
    for P in P_list:
        NP, TP_weak = weak_data[P]
        work_ratio = (NP * NP) / (N * N)
        sp_scaled = P * (1.0 / work_ratio) * (T1_weak / TP_weak) if TP_weak > 0 else 0.0
        weak_rows.append(f"{P}\t{NP}x{NP}\t{TP_weak:.6f}\t{sp_scaled:.4f}")

    os.makedirs("results", exist_ok=True)
    tsv_path = os.path.join("results", "scaling_metrics.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("# --- STRONG SCALING ---\n")
        f.write("\n".join(strong_rows) + "\n\n")
        f.write("# --- WEAK SCALING ---\n")
        f.write("\n".join(weak_rows) + "\n")

    print(f"[SUCCESS] Scaling metrics written to {tsv_path}")

    flops_per_elem = 5.0 * 3  
    bytes_per_elem = 8.0      
    arithmetic_intensity = flops_per_elem / bytes_per_elem 
    
    roofline_summary = {
        "student_id": student_id,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "execution_regime": "Memory-Bandwidth Bound",
        "rationale": "Stencil updates stream large arrays through cache/DRAM with low arithmetic intensity, saturating memory bandwidth before compute units peak.",
        "max_theoretical_speedup_amdahl": 1.0 / strong_metrics[-1][4] if len(strong_metrics) > 1 and strong_metrics[-1][4] < 1.0 else 0.0
    }

    json_path = os.path.join("results", "roofline_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(roofline_summary, f, indent=4)

    print(f"[SUCCESS] Roofline summary written to {json_path}")

if __name__ == "__main__":
    main()