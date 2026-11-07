import os
from datetime import timedelta

import torch
import torch.distributed as dist


def run_worker():
    print(f"torch.distributed.is_initialized() {torch.distributed.is_initialized()}")
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 初始化进程组，不再手动传 rank/world_size
    dist.init_process_group(backend="gloo",  # 指定底层通信库
                            init_method="env://", # 告诉进程“去哪里找彼此”。默认是 "env://"，即从环境变量 MASTER_ADDR、MASTER_PORT 读取会合地址。也可以写成 "tcp://192.168.153.11:29500" 直接指定
                            world_size=4, # world_size 总进程数 所有进程必须传相同的值
                            rank=rank,  #当前进程的全局编号
                            timeout=timedelta(seconds=10))  # GPU 环境请换 "nccl"

    tensor = torch.tensor([rank + 1.0], dtype=torch.float32)
    print(f"[Rank {rank}] 原始 tensor: {tensor.item()}", flush=True)

    work: torch.distributed.Work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    # ... 你的计算逻辑 ...
    work.wait()
    print(f"[Rank {rank}] All-Reduce 后: {tensor.item()}", flush=True)

    print(f"torch.distributed.is_initialized() {torch.distributed.is_initialized()}")
    dist.destroy_process_group()

"""
第一个和第二个节点分别运行
torchrun   --nnodes=2   --nproc_per_node=2   --node_rank=1   --master_addr=192.168.153.11   --master_port=29500   torch_demo.py
torchrun   --nnodes=2   --nproc_per_node=2   --node_rank=1   --master_addr=192.168.153.11   --master_port=29500   torch_demo.py

"""
if __name__ == "__main__":
    run_worker()
