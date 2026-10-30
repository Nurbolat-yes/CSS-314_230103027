import os
import sys
import json
import platform
import subprocess
import argparse
import psutil
from salt import get_student_params

def get_cpu_info():
    info = {
        "model_name": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_threads": psutil.cpu_count(logical=True),
        "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }
    
    if platform.system() == "Windows":
        try:

            cmd = "wmic cpu get Name, L2CacheSize, L3CacheSize /format:list"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")
            for line in output.split("\r\n"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip()
                    if key == "Name" and val:
                        info["model_name"] = val
                    elif key == "L2CacheSize" and val:
                        info["l2_cache"] = f"{val} KB"
                    elif key == "L3CacheSize" and val:
                        info["l3_cache"] = f"{val} KB"
        except Exception:
            pass
            
    return info

def main():
    parser = argparse.ArgumentParser(description="Profile environment and generate student salted parameters.")
    parser.add_argument("--student-id", required=True, help="8-digit Student ID")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("src", exist_ok=True)

    student_params = get_student_params(args.student_id)

    profile = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "kernel": platform.uname().release,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "student_params": student_params
    }

    output_path = os.path.join("results", "host_profile.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Host profile successfully written to {output_path}")
    print(f"[SUCCESS] Student parameters generated: {student_params}")

if __name__ == "__main__":
    main()