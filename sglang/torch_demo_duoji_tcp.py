# multi_node_tcp_demo.py
import os
import sys
import multiprocessing as mp
from datetime import timedelta

import torch
import torch.distributed as dist


def run_worker(local_rank, world_size, master_addr, master_port, node_rank, nproc_per_node):
    # 核心：计算全局 rank
    global_rank = node_rank * nproc_per_node + local_rank

    # 使用 tcp:// 作为 init_method
    init_method = f"tcp://{master_addr}:{master_port}"

    print(f"[Node {node_rank}][Local {local_rank}][Global {global_rank}] "
          f"正在使用 {init_method} 初始化...", flush=True)

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=global_rank,
        timeout=timedelta(seconds=60),  # 多机握手时间可以给长一点
    )

    tensor = torch.tensor([global_rank + 1.0], dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Global {global_rank}] All-Reduce 结果: {tensor.item()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    # 从命令行参数读取本机配置
    # 用法: python multi_node_tcp_demo.py <node_rank> <nproc_per_node> <master_addr> <master_port>
    node_rank = int(sys.argv[1])
    nproc_per_node = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = sys.argv[4]

    # 总进程数 = 假设有 2 个节点
    world_size = 2 * nproc_per_node

    mp.set_start_method("spawn", force=True)

    procs = []
    for local_rank in range(nproc_per_node):
        p = mp.Process(
            target=run_worker,
            args=(local_rank, world_size, master_addr, master_port, node_rank, nproc_per_node),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # 检查异常
    for local_rank, p in enumerate(procs):
        if p.exitcode != 0:
            print(f"Local rank {local_rank} 异常退出，exitcode={p.exitcode}")