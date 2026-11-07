import sys
import multiprocessing as mp
from datetime import timedelta

import torch
import torch.distributed as dist


def run_worker(local_rank, world_size, master_addr, master_port, node_rank, nproc_per_node):
    # 全局 rank = 节点编号 * 每节点进程数 + 本机进程序号
    global_rank = node_rank * nproc_per_node + local_rank
    init_method = f"tcp://{master_addr}:{master_port}"

    print(f"[Global {global_rank}] 初始化 WORLD...", flush=True)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=global_rank,
        timeout=timedelta(seconds=60),
    )

    # ---------------------------------------------------------------
    # 关键：所有进程都必须调用 new_group，且调用顺序、ranks 完全一致
    # 这里切出两个 TP 子组：[0,1] 和 [2,3]
    # ---------------------------------------------------------------
    # tp_group_ranks = [[0, 1], [2, 3]]
    tp_group_ranks = [[0, 1], [1, 3]]


    # 先为每个子组创建 group 句柄（所有 rank 都参与创建）
    tp_groups = []
    for ranks in tp_group_ranks:
        g = dist.new_group(ranks=ranks)
        tp_groups.append(g)

    # 判断当前进程属于哪个 TP 组
    my_tp_group = None
    my_tp_rank = None
    for idx, ranks in enumerate(tp_group_ranks):
        if global_rank in ranks:
            my_tp_group = tp_groups[idx]
            my_tp_rank = ranks.index(global_rank)
            break

    print(f"[Global {global_rank}] 属于 TP 组 {tp_group_ranks[[i for i,r in enumerate(tp_group_ranks) if global_rank in r][0]]}, "
          f"组内 rank={my_tp_rank}", flush=True)

    # ---------------------------------------------------------------
    # 在子组内做 all_reduce：只有同组进程参与，不同组之间互不影响
    # ---------------------------------------------------------------
    tensor = torch.tensor([global_rank + 1.0], dtype=torch.float32)
    print(f"[Global {global_rank}] 子组 all_reduce 前: {tensor.item()}", flush=True)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=my_tp_group)

    print(f"[Global {global_rank}] 子组 all_reduce 后: {tensor.item()}", flush=True)

    # 对照：在 WORLD 组内再做一次 all_reduce，验证全局通信仍然可用
    world_tensor = torch.tensor([global_rank + 1.0], dtype=torch.float32)
    dist.all_reduce(world_tensor, op=dist.ReduceOp.SUM)
    print(f"[Global {global_rank}] WORLD all_reduce 后: {world_tensor.item()}", flush=True)

    dist.destroy_process_group()

"""
python multi_node_new_group_demo.py 0 2 192.168.153.11 29501
python multi_node_new_group_demo.py 1 2 192.168.153.11 29501
"""
if __name__ == "__main__":
    # 用法: python multi_node_new_group_demo.py <node_rank> <nproc_per_node> <master_addr> <master_port>
    node_rank = int(sys.argv[1])
    nproc_per_node = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = sys.argv[4]

    # 2 个节点
    nnodes = 2
    world_size = nnodes * nproc_per_node

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

    for local_rank, p in enumerate(procs):
        if p.exitcode != 0:
            print(f"Local rank {local_rank} 异常退出，exitcode={p.exitcode}")