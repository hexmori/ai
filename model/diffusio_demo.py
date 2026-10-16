import torch
from diffusers import DiffusionPipeline
from PIL import Image

# 1. 加载预训练模型
# 使用 sd-turbo 模型，它体积小（约 1.7GB），生成速度快，支持 1~4 步快速生成
model_id = "stabilityai/sd-turbo"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # 用半精度减少显存占用
    variant="fp16",
)

# 2. 将模型移到 GPU（如果没有 GPU，用 "cpu" 也可以，只是慢很多）
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# 3. 定义你的文字提示
prompt = "A cute orange cat wearing sunglasses on a sunny beach, photorealistic"

# 4. 调用管道生成图像
# num_inference_steps：去噪迭代次数。sd-turbo 通常 1~4 步就能出效果[citation:2]
# guidance_scale：提示词遵循度。sd-turbo 对 guidance 较敏感，可尝试 0~2.5
image = pipe(
    prompt=prompt,
    num_inference_steps=4,       # 去噪步数
    guidance_scale=0.0,          # sd-turbo 推荐 0.0 或很低的值
).images[0]

# 5. 保存图像
image.save("generated_cat.png")


print("图像已保存为 generated_cat.png")