"""
模拟 torchgen 的核心原理：YAML 声明 -> 代码生成
"""
import yaml
from dataclasses import dataclass
from typing import List


# 1. 模拟 torchgen 的“数据模型”（对应 torchgen.model.NativeFunction）
@dataclass
class NativeFunction:
    name: str
    args: List[str]
    return_type: str


# 2. 模拟“生成器”（对应 torchgen.dest 下的各种 Generator）
def generate_registration_code(func: NativeFunction) -> str:
    """为单个算子生成 Dispatcher 注册代码的字符串（模拟 C++ 输出）"""
    # 实际 torchgen 会生成类似 m.impl("add.Tensor", TORCH_FN(add_cpu_kernel)); 的 C++
    # 这里用 Python 字符串模拟这个“写代码”的动作
    arg_str = ", ".join(func.args)
    return f"""
// 为算子 {func.name} 生成的注册存根
// 签名: {func.return_type} {func.name}({arg_str})
void register_{func.name}() {{
    // 实际 C++: m.impl("{func.name}", TORCH_FN({func.name}_kernel));
    std::cout << "Registering {func.name}" << std::endl;
}}
""".strip()


# 3. 模拟主流程（对应 torchgen/gen.py 的 main 逻辑）
def run_torchgen_like_flow(yaml_content: str):
    # 步骤 A: 解析 YAML（实际对应 parse_native_yaml）
    data = yaml.safe_load(yaml_content)

    generated_files = {}

    # 步骤 B: 遍历每个函数定义
    for item in data.get("functions", []):
        # 构建数据模型
        func = NativeFunction(
            name=item["name"],
            args=item["args"],
            return_type=item["return"]
        )

        # 步骤 C: 调用生成器，得到代码字符串
        code = generate_registration_code(func)
        generated_files[f"Register_{func.name}.cpp"] = code

    return generated_files


# ================= 运行示例 =================
if __name__ == "__main__":
    # 这是我们的“native_functions.yaml”迷你版
    mini_yaml = """
functions:
  - name: "my_add"
    args: ["Tensor a", "Tensor b"]
    return: "Tensor"
  - name: "my_relu"
    args: ["Tensor x"]
    return: "Tensor"
"""

    # 执行生成流程
    outputs = run_torchgen_like_flow(mini_yaml)

    # 打印“生成”的文件内容
    print("=== torchgen 模拟生成结果 ===\n")
    for filename, content in outputs.items():
        print(f"--- 生成文件: {filename} ---")
        print(content)
        print()

    import torch

    ops = torch._C._dispatch_get_all_op_names()

    print("算子数量:", len(ops))

    for op in sorted(ops):
        print(op)