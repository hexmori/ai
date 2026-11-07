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

    # ---------------------------------------------------------------
    # 所有进程都必须调用 new_group，且顺序、ranks 完全一致
    # 创建一个只包含全局 rank 2 和 3 的子组
    # ---------------------------------------------------------------
    subgroup_ranks = [2, 3]
    subgroup = dist.new_group(ranks=subgroup_ranks)

    # ---------------------------------------------------------------
    # 只有子组成员才调用带 group 的 send/recv
    # 非成员进程调用 send/recv 会触发 get_group_rank 的 ValueError
    # ---------------------------------------------------------------
    if rank == 2:
        # 全局 rank 2 = 子组内局部 rank 0，负责发送
        tensor = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        print(f"[Global {rank}][Group 0] 准备发送: {tensor.tolist()}", flush=True)
        # dst=1 是子组内局部编号，指向全局 rank 3
        dist.send(tensor=tensor, dst=1, group=subgroup)
        print(f"[Global {rank}][Group 0] 发送完成", flush=True)

    elif rank == 3:
        # 全局 rank 3 = 子组内局部 rank 1，负责接收
        recv_tensor = torch.zeros(3, dtype=torch.float32)
        print(f"[Global {rank}][Group 1] 等待接收...", flush=True)
        # src=0 是子组内局部编号，指向全局 rank 2
        dist.recv(tensor=recv_tensor, src=0, group=subgroup)
        print(f"[Global {rank}][Group 1] 接收完成: {recv_tensor.tolist()}", flush=True)

    else:
        # 全局 rank 0 和 1：只参与了 new_group 创建，不参与 send/recv
        print(f"[Global {rank}] 不在子组内，仅参与 new_group 创建，跳过 send/recv", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    world_size = 4
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