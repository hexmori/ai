import torch

if __name__ == "__main__":
    """
    它的核心作用是：你告诉它你想操作什么设备（比如 "cuda"、"xpu"、"npu"），
    它就返回对应的 PyTorch 子模块（如 torch.cuda、torch.xpu、torch.npu），让你能用统一的代码调用不同硬件后端的 API
    """
    device_module = torch.get_device_module()
    event = device_module.Event()


    event.record()

    event.synch