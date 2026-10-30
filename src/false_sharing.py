import os
import time
import threading
import numpy as np

def worker_unpadded(shared_arr, worker_id, iterations):
    for _ in range(iterations):
        shared_arr[worker_id] += 1

def worker_padded(shared_arr, worker_id, iterations):
    idx = worker_id * 8
    for _ in range(iterations):
        shared_arr[idx] += 1

def worker_privatized(shared_arr, worker_id, iterations):
    local_val = 0
    for _ in range(iterations):
        local_val += 1
    shared_arr[worker_id * 8] = local_val

def run_benchmark(mode, P_list, iterations=5_000_000):
    results = {}
    
    # Замер T1 (1 поток)
    shared_arr = np.zeros(64, dtype=np.int64)
    start = time.perf_counter()
    if mode == "unpadded":
        t = threading.Thread(target=worker_unpadded, args=(shared_arr, 0, iterations))
    elif mode == "padded":
        t = threading.Thread(target=worker_padded, args=(shared_arr, 0, iterations))
    else:
        t = threading.Thread(target=worker_privatized, args=(shared_arr, 0, iterations))
    t.start()
    t.join()
    end = time.perf_counter()
    T1 = end - start
    results[1] = T1

    # Запуск для P потоков
    for P in P_list:
        if P == 1:
            continue
        shared_arr = np.zeros(P * 8 if mode != "unpadded" else P, dtype=np.int64)
        
        threads = []
        start = time.perf_counter()
        for i in range(P):
            if mode == "unpadded":
                t = threading.Thread(target=worker_unpadded, args=(shared_arr, i, iterations))
            elif mode == "padded":
                t = threading.Thread(target=worker_padded, args=(shared_arr, i, iterations))
            else:
                t = threading.Thread(target=worker_privatized, args=(shared_arr, i, iterations))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
        end = time.perf_counter()
        results[P] = end - start
        
    return results

def main():
    iterations = 5_000_000
    P_list = [1, 2, 4, 8]
    
    print(f"[INFO] Running fast False Sharing benchmark (Threading) with K = {iterations:,} iterations...")
    
    modes = ["unpadded", "padded", "privatized"]
    raw_data = {}
    
    for mode in modes:
        print(f"[INFO] Benchmarking mode: {mode.upper()}...")
        raw_data[mode] = run_benchmark(mode, P_list, iterations)

    tsv_rows = ["mode\tP\truntime_s\tspeedup_SP\tkarp_flatt_eP"]
    
    for mode in modes:
        T1 = raw_data[mode][1]
        for P in P_list:
            TP = raw_data[mode][P]
            SP = T1 / TP if TP > 0 else 0.0
            
            if P == 1:
                eP = 0.0
            else:
                eP = ((P * TP / T1) - 1) / (P - 1)
                
            tsv_rows.append(f"{mode}\t{P}\t{TP:.6f}\t{SP:.4f}\t{eP:.4f}")
            print(f"Mode: {mode:10s} | P={P} | Time: {TP:7.4f}s | Speedup: {SP:6.2f}x | Karp-Flatt e(P): {eP:6.4f}")

    os.makedirs("results", exist_ok=True)
    tsv_path = os.path.join("results", "false_sharing_karp_flatt.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tsv_rows) + "\n")

    print(f"[SUCCESS] False sharing & Karp-Flatt results written to {tsv_path}")

if __name__ == "__main__":
    main()