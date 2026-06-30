import psutil
import time
import argparse
import json
import subprocess
import signal
import sys
import matplotlib.pyplot as plt

stats = []
output_json = ""
output_jpg = ""

def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_util = 0
        mem_used = 0
        mem_total = 0
        for line in lines:
            parts = line.split(', ')
            if len(parts) == 3:
                gpu_util = max(gpu_util, float(parts[0]))
                mem_used += float(parts[1])
                mem_total += float(parts[2])
        return gpu_util, mem_used, mem_total
    except Exception:
        return 0, 0, 0

def save_and_exit(signum, frame):
    global output_json, output_jpg, stats
    
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    if stats:
        times = [s['time_s'] for s in stats]
        cpu = [s['cpu_percent'] for s in stats]
        ram = [s['ram_percent'] for s in stats]
        gpu = [s['gpu_util_percent'] for s in stats]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, cpu, label='CPU (%)')
        plt.plot(times, ram, label='RAM (%)')
        plt.plot(times, gpu, label='GPU Utilization (%)')
        plt.xlabel('Time (s)')
        plt.ylabel('Utilization (%)')
        plt.title('System Resources Usage')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_jpg)
        plt.close()
    
    sys.exit(0)

def main():
    global output_json, output_jpg, stats
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--output_jpg', type=str, required=True)
    args = parser.parse_args()
    
    output_json = args.output_json
    output_jpg = args.output_jpg
    
    signal.signal(signal.SIGTERM, save_and_exit)
    signal.signal(signal.SIGINT, save_and_exit)
    
    # Initialize CPU percent
    psutil.cpu_percent(interval=0.1)
    
    start_time = time.time()
    try:
        while True:
            current_time = time.time() - start_time
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used_gb = ram.used / (1024**3)
            
            gpu_util, gpu_mem_used, gpu_mem_total = get_gpu_stats()
            gpu_mem_used_gb = gpu_mem_used / 1024
            
            stats.append({
                'time_s': current_time,
                'cpu_percent': cpu_percent,
                'ram_percent': ram_percent,
                'ram_used_gb': ram_used_gb,
                'gpu_util_percent': gpu_util,
                'gpu_mem_used_gb': gpu_mem_used_gb
            })
            time.sleep(1)
    except Exception as e:
        print(f"Error in monitor: {e}")
        save_and_exit(None, None)

if __name__ == '__main__':
    main()
