import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn

if __name__ == "__main__":

    # 1. 加载 tokenizer（分词器）
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("xxx")
    print(tokenizer.all_special_ids)
    print(tokenizer.all_special_tokens)
    print("xxx")
    # 2. 加载模型（不使用 vLLM）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # 自动放 GPU
    )

    prompt = "Hello, my name is"

    # ------------------------------
    # ✔ 3. 分词（tokenize）
    # ------------------------------
    inputs = tokenizer(prompt, return_tensors="pt")

    # input_ids 是模型真正吃的输入
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    print("input_ids:", input_ids)
    print("对应的词：", tokenizer.convert_ids_to_tokens(input_ids[0]))

    # ------------------------------
    # ✔ 4. 推理（prefill + decode）
    # ------------------------------
    # prefill 阶段：模型吃所有输入
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # outputs 包含：
    # logits: [batch, seq_len, vocab_size]
    # logits = outputs.logits
    logits = outputs[0]
    # mx.argmax(logits[:, -1, :], axis=-1)
    print("")

    # 取最后一个 token 的 logits（下一步的预测）
    next_token_logits = logits[:, -1, :]

    # 选 top1（贪心）
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    print("预测的下一个 token ID:", next_token_id.item())
    print("预测的 token:", tokenizer.decode(next_token_id))

    # ------------------------------
    # ✔ 5. 手动 decode 一个 token（模拟 streaming）
    # ------------------------------
    generated = input_ids
    for i in range(20):  # 生成 20 个 token
        with torch.no_grad():
            out = model(input_ids=generated)

        next_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0)

        generated = torch.cat([generated, next_token], dim=1)

    result = tokenizer.decode(generated[0])
    print("最终生成结果：")
    print(result)

    nn.Module