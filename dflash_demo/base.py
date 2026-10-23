import torch.nn as nn
import torch
import numpy as np

vocab_size = 5
codebook_dim = 16
embedding = nn.Embedding(vocab_size, codebook_dim)
print(embedding)

# 单个索引
idx1 = torch.tensor(2)
vec1 = embedding(idx1)  # 输出形状: (16,)

# 一批索引
idx2 = torch.tensor([0, 2, 4])
vecs2 = embedding(idx2)  # 输出形状: (3, 16)

print("=====================")

m = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))  # 共享内存
print(m)
print(m[:,2])
