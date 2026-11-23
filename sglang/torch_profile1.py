import torch
print(f"torch.cuda.is_available() {torch.cuda.is_available()}")
print(f"torch.version.cuda is not None {torch.version.cuda is not None}")


import torch
import torch_npu

s1 = torch.npu.Stream()
s2 = torch.npu.Stream()

print("s1:", s1)
print("s1 id:", id(s1))
print("s1 npu_stream:", getattr(s1, "npu_stream", None))

print("s2:", s2)
print("s2 id:", id(s2))
print("s2 npu_stream:", getattr(s2, "npu_stream", None))

with torch.npu.stream(s1):
    x = torch.randn(1024, 1024, device="npu")
    y = x @ x

with torch.npu.stream(s2):
    z = y @ y

torch.npu.synchronize()