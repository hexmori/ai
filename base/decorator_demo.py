# registry_demo.py

# ========== 1. 注册表本身 ==========
_HANDLERS = {}

def register_handler(name):
    """返回一个装饰器，把被装饰的函数以 name 为键存进 _HANDLERS。"""
    def decorator(fn):
        _HANDLERS[name] = fn
        return fn
    return decorator


def build_handler(name):
    """根据名字查注册表，找到工厂函数并调用它创建实例。"""
    if name not in _HANDLERS:
        raise ValueError(f"未知的 handler: {name!r}，已注册: {list(_HANDLERS)}")
    factory = _HANDLERS[name]
    return factory()


# ========== 2. 各个后端用装饰器注册自己 ==========
class CudaHandler:
    def __init__(self):
        self.device = "cuda"
    def handle(self, x):
        return f"[CUDA] 处理 {x}"


class CpuHandler:
    def __init__(self):
        self.device = "cpu"
    def handle(self, x):
        return f"[CPU] 处理 {x}"


class AscendHandler:
    def __init__(self):
        self.device = "ascend"
    def handle(self, x):
        return f"[Ascend] 处理 {x}"


# 这行在模块被 import 时就会执行，把 create_cuda_handler 注册进字典。不需要手动调用注册函数。
@register_handler("cuda")
def create_cuda_handler():
    return CudaHandler()


@register_handler("cpu")
def create_cpu_handler():
    return CpuHandler()


@register_handler("ascend")
def create_ascend_handler():
    return AscendHandler()


# ========== 3. 使用：根据名字创建 ==========
if __name__ == "__main__":
    print("已注册的 handler:", list(_HANDLERS.keys()))

    for name in ["cuda", "cpu", "ascend"]:
        handler = build_handler(name)
        print(handler.handle(42))

    # 未知名字会报错，并列出已注册的
    try:
        build_handler("tpu")
    except ValueError as e:
        print("错误:", e)