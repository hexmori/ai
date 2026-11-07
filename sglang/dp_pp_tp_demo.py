from typing import Tuple


def _calculate_rank_ranges(
        nnodes: int, pp_size: int, tp_size: int, node_rank: int
) -> Tuple[range, range, int, int]:
    """Calculate pp_rank_range and tp_rank_range for a given node.

    Args:
        nnodes: Total number of nodes.
        pp_size: Pipeline parallel size.
        tp_size: Tensor parallel size.
        node_rank: The rank of the node to compute ranges for.

    Returns:
        A tuple of (pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node):
        - pp_rank_range: range of pipeline-parallel ranks assigned to this node.
        - tp_rank_range: range of tensor-parallel ranks assigned to this node.
        - pp_size_per_node: number of PP ranks per node.
        - tp_size_per_node: number of TP ranks per node.
    """
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )

    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node


nnodes = 2
pp_size = 8
tp_size = 2
node_rank = 1
pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = _calculate_rank_ranges(nnodes=nnodes,
                                                                                          pp_size=pp_size,
                                                                                          tp_size=tp_size,
                                                                                          node_rank=node_rank)
print(f"pp_rank_range {pp_rank_range}")
print(f"tp_rank_range {tp_rank_range}")
print(f"pp_size_per_node {pp_size_per_node}")
print(f"tp_size_per_node {tp_size_per_node}" )
