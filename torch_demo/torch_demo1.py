import torch

tensor = torch.tensor([1.0], dtype=torch.float)
print(f"tensor {tensor}")
lists: list[int] = [1, 4, 1, 2]
torch.tensor(lists, dtype=torch.int64)
