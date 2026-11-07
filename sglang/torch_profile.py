import torch
import time

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    with_stack=True
) as prof:

    with torch.profiler.record_function("scheduler.run_batch"):
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        c = torch.matmul(a, b)

    with torch.profiler.record_function("scheduler.process_batch_result"):
        time.sleep(0.5)

# 导出 Chrome Trace
prof.export_chrome_trace("cpu_profile.json")

# 输出算子统计信息
print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20
    )
)

print("Profile 已生成")