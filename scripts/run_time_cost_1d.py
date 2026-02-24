import subprocess
from multiprocessing import Pool

# 要运行的 L 值
L_list = [20, 40, 60, 80, 100, 120, 140]

# 最大并发数（控制 CPU 占用）
MAX_PROCESSES = min(len(L_list), 8)

# 单个任务运行函数
def run_task(L):
    cmd = ["python", "vmc_spinful_1d_run_input_L.py", str(L)]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 打印运行结果
    print(f"L={L} finished. Return code: {result.returncode}")
    if result.stdout:
        print(f"[stdout L={L}]\n{result.stdout.decode()}")
    if result.stderr:
        print(f"[stderr L={L}]\n{result.stderr.decode()}")

# 并行执行
if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)  # 为稳妥起见

    with Pool(processes=MAX_PROCESSES) as pool:
        pool.map(run_task, L_list)
