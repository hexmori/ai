import torch
import time

# ============================================================
# 1. 定义一个由多个小算子组成的函数
#    对应 SGLang 里那种 "RMSNorm + scale + shift" 式的算子链
# ============================================================
def fn_eager(x, weight, bias):
    # 算子链: 归一化 -> 缩放 -> 平移 -> 激活
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)   # 归一化
    x = x * weight                                   # 缩放
    x = x + bias                                     # 平移
    x = torch.relu(x)                                # 激活
    return x


# ============================================================
# 2. 用 torch.compile 装饰同一个函数
#    dynamic=True 支持动态 shape（对应多轮对话中变化的 seq_len）
# ============================================================
@torch.compile(dynamic=True)
def fn_compiled(x, weight, bias):
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    x = x * weight
    x = x + bias
    x = torch.relu(x)
    return x


# ============================================================
# 3. 准备输入
# ============================================================
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

batch, seq, hidden = 8, 512, 1024
x = torch.randn(batch, seq, hidden, device=device)
weight = torch.randn(hidden, device=device)
bias = torch.randn(hidden, device=device)


# ============================================================
# 4. 正确性验证：编译后的结果必须和 eager 一致
# ============================================================
out_eager = fn_eager(x, weight, bias)
out_compiled = fn_compiled(x, weight, bias)

max_diff = (out_eager - out_compiled).abs().max().item()
print(f"\n[正确性] eager 与 compiled 的最大误差: {max_diff:.2e}")
assert max_diff < 1e-5, "编译结果与 eager 不一致!"
print("[正确性] 通过 ✓")


# ============================================================
# 5. 性能对比
# ============================================================
def benchmark(fn, x, weight, bias, warmup=10, iters=100):
    # 预热（第一次调用会触发编译，必须排除）
    for _ in range(warmup):
        fn(x, weight, bias)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(x, weight, bias)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000  # 毫秒/次


# 先单独调用一次 compiled，触发实际编译
_ = fn_compiled(x, weight, bias)
if device == "cuda":
    torch.cuda.synchronize()

t_eager = benchmark(fn_eager, x, weight, bias)
t_compiled = benchmark(fn_compiled, x, weight, bias)

print(f"\n[性能] eager     : {t_eager:.4f} ms/次")
print(f"[性能] compiled  : {t_compiled:.4f} ms/次")
print(f"[性能] 加速比    : {t_eager / t_compiled:.2f}x")


# ============================================================
# 6. 查看编译后到底融合成了几个内核
# ============================================================
print("\n[内核数量] eager 模式下, 上述函数会启动多个独立内核")
print("[内核数量] compiled 模式下, Inductor 会把它们融合成更少的内核")

# 尝试打印 Inductor 生成的代码（需要设置环境变量才可见）
import os
if os.environ.get("TORCH_COMPILE_DEBUG"):
    print("\n提示: 设置 TORCH_COMPILE_DEBUG=1 可看到生成的 Triton 代码")