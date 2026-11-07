import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_worker(rank, world_size):
    # 1. 初始化进程组 (Windows 上必须使用 gloo)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'  # Windows 上回环接口名通常是 'lo' 或 'Loopback'
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method="env://?use_libuv=False"
    )

    # 准备数据：rank 0 的 tensor 是 1，rank 1 的 tensor 是 2
    tensor = torch.tensor([rank + 1.0], dtype=torch.float32)
    print(f"[Rank {rank}] 准备就绪，原始 tensor: {tensor.item()}")

    # 2. 发起【异步】 All-Reduce 操作
    # async_op=True 会立即返回一个 Work 对象，而不会阻塞当前线程
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

    # --- 在等待通信完成的同时，去做一些其他计算（模拟算力重叠） ---
    print(f"[Rank {rank}] 通信已在后台进行中... 现在去算一点别的东西。")
    local_compute_result = 0
    for i in range(1000000):
        local_compute_result += i

    # 3. 必须调用 wait() 确保通信完成
    # 如果不调用 wait()，直接读取 tensor，可能会读到未更新完的数据（尤其是在 CUDA 上）
    work.wait()
    print(f"[Rank {rank}] 通信完成！")

    # 4. 此时 tensor 已经被更新（1 + 2 = 3.0）
    print(f"[Rank {rank}] All-Reduce 后的 tensor: {tensor.item()}")

    # 5. 销毁进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 2
    # 一个机器上启动两个子进程运行
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)