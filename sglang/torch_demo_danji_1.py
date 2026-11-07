# tcp_init_demo.py
import os
import multiprocessing as mp          # ← 换成标准库
from datetime import timedelta

import torch
import torch.distributed as dist


def run_worker(rank, world_size, master_addr, master_port):
    init_method = f"tcp://{master_addr}:{master_port}"
    print(f"[Rank {rank}] 尝试使用 {init_method} 初始化...", flush=True)

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=30),
    )

    tensor = torch.tensor([rank + 1.0], dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] All-Reduce 完成，结果: {tensor.item()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    # 和 SGLang 一样，显式指定 spawn 启动方式
    mp.set_start_method("spawn", force=True)

    world_size = 2
    master_addr = "127.0.0.1"
    master_port = "29501"

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_worker,
            args=(rank, world_size, master_addr, master_port),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # 检查是否有子进程异常退出
    for rank, p in enumerate(procs):
        if p.exitcode != 0:
            print(f"[Rank {rank}] 异常退出，exitcode={p.exitcode}")