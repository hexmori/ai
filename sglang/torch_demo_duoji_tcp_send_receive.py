import multiprocessing as mp
from datetime import timedelta

import torch
import torch.distributed as dist


def run_worker(rank, world_size, master_addr, master_port):
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=60),
    )

    if rank == 0:
        # ---- 发送方 ----
        # 模拟 stage 0 的计算结果
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        print(f"[Rank 0] 准备发送: {tensor.tolist()}", flush=True)

        # 同步发送给 rank 1。注意：send 会阻塞，直到 rank 1 调 recv
        dist.send(tensor=tensor, dst=1)
        print(f"[Rank 0] 发送完成", flush=True)
    else:
        # ---- 接收方 ----
        # 必须先分配好接收张量，shape/dtype 要和发送方一致
        recv_tensor = torch.zeros(4, dtype=torch.float32)
        print(f"[Rank 1] 等待接收...", flush=True)

        # 同步接收来自 rank 0 的数据。recv 会阻塞直到数据到达
        dist.recv(tensor=recv_tensor, src=0)
        print(f"[Rank 1] 接收完成: {recv_tensor.tolist()}", flush=True)

        # 模拟 stage 1 的后续计算
        result = recv_tensor * 2
        print(f"[Rank 1] 后续计算结果: {result.tolist()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
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

    for rank, p in enumerate(procs):
        if p.exitcode != 0:
            print(f"[Rank {rank}] 异常退出，exitcode={p.exitcode}")