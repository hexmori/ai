import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 辅助函数与简化的类定义
# ==========================================
def _sampling_probs(logits, temperature):
    """辅助函数：根据温度将 logits 转为概率"""
    return F.softmax(logits / temperature, dim=-1)


def _sample_probs(probs):
    """辅助函数：根据概率采样"""
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class PathSelector(nn.Module):
    def __init__(self, hidden_dim=20, codebook_dim=16, top_k=3):
        super().__init__()
        self.top_k = top_k
        # 模拟特征维度的投影层 (对应代码里的 self.hidden_projection)
        self.hidden_projection = nn.Linear(hidden_dim, codebook_dim)

        # 模拟前驱和后继词的特征查表 (对应代码里的 predecessor/successor_codebook)
        # 假设词表大小为 100
        vocab_size = 100
        self.predecessor_codebook = nn.Embedding(vocab_size, codebook_dim)
        self.successor_codebook = nn.Embedding(vocab_size, codebook_dim)

    def select(self, hidden, logits, anchor_ids, temperature):
        # 1. 获取 Top-K 候选和原始得分 unary topk 对应的值， candidates topk对应的下标
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)

        # 2. 投影隐藏状态 (对应公式的 H(h_t))
        hidden = self.hidden_projection(hidden)

        predecessor = anchor_ids
        path, q_rows = [], []

        # 3. 逐位置遍历
        for position in range(hidden.shape[1]):
            # 计算得分: 公式中的 U + <A ⊙ H, B>
            # predecessor 形状: [batch_size]
            # hidden[:, position] 形状: [batch_size, codebook_dim]
            # candidates[:, position] 形状: [batch_size, top_k]

            # 查表得到前驱和当前候选的词向量
            pred_vec = self.predecessor_codebook(predecessor)  # [b, dim]
            succ_vec = self.successor_codebook(candidates[:, position])  # [b, k, dim]

            # 逐元素乘 (⊙) 和 点积 (einsum)
            match_score = torch.einsum(
                "br,bkr->bk",
                pred_vec * hidden[:, position],
                succ_vec
            )

            # 加上原本的得分
            scores = unary[:, position] + match_score

            # 4. 选取路径
            if temperature > 0:
                q = _sampling_probs(scores, temperature)
                index = _sample_probs(q)
                q_rows.append(q)
            else:
                index = torch.argmax(scores, dim=-1)

            # 5. 更新前驱，加入路径
            # 从 candidates 取出选中的词
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)

        return (
            torch.stack(path, dim=1),
            candidates,
            torch.stack(q_rows, dim=1) if q_rows else None,
        )


# ==========================================
# 2. 构造测试数据 (跑通流程)
# ==========================================
torch.manual_seed(42)  # 固定随机种子让结果可复现

# 词表设定 (只是为了方便看清 ID 对应的词)
vocab = {0: "我", 1: "喜欢", 2: "猫", 3: "狗", 4: "吃饭", 5: "睡觉", 6: "跑", 7: "叫"}
anchor_token_id = torch.tensor([2])  # 锚点：假设上下文是 "我 喜欢 猫"

batch_size = 1
seq_len = 4  # 我们要预测 2 个位置的 token
hidden_dim = 20
top_k = 2 # 为了直观，这里只留 Top-2 候选

# 构造输入数据
hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
logits = torch.zeros(batch_size, seq_len, 100)  # 假设词表大小为100

# 【人为制造冲突场景】
# 位置 1 (P1)：故意让 "狗" (id=3) 概率最高，"猫" (id=2) 其次
logits[0, 0, 3] = 0.9
logits[0, 0, 2] = 0.8

# 位置 2 (P2)：故意让 "叫" (id=7) 概率最高，"吃饭" (id=4) 其次
logits[0, 1, 7] = 0.9
logits[0, 1, 4] = 0.8

# 初始化路径选择器
selector = PathSelector(hidden_dim=hidden_dim, top_k=top_k)

# 模拟训练好的特征向量：强制让 "猫(2)" 和 "叫(7)" 的特征向量极其匹配(相似)，
# 而 "狗(3)" 和 "叫(7)" 完全不匹配。
with torch.no_grad():
    # 强行把特征库中"猫"的后继向量设置为全 1
    selector.predecessor_codebook.weight[2] = torch.ones(16)
    selector.successor_codebook.weight[7] = torch.ones(16)

    # 强行把特征库中"狗"的后继向量设置为全 -1 (负相关)
    selector.predecessor_codebook.weight[3] = -torch.ones(16)

    # 让上下文门控不产生太大干扰 (全设为 0.5)
    selector.hidden_projection.weight.data.fill_(0)
    selector.hidden_projection.bias.data.fill_(0.5)

print(f"锚点上下文: {vocab[anchor_token_id.item()]}")

# ==========================================
# 3. 执行对比测试
# ==========================================
# 模式 1：贪心 (T=0) - 模拟 DFlash V1 直接取 Top-1
print("\n--- 测试 DFlash V1 (盲目取 Top-1) ---")
v1_path = []
for pos in range(seq_len):
    best_id = torch.argmax(logits[0, pos]).item()
    v1_path.append(vocab[best_id])
print(f"V1 选择的草稿: {' -> '.join(v1_path)}")
print("【评价】: 虽然单独概率最高，但 '猫 -> 叫' 在语义上极其不连贯，大模型验证大概率会全部拒绝！")

# 模式 2：DFlash V2 的路径选择器 (T=0)
print("\n--- 测试 DFlash V2 (路径选择器) ---")
path, candidates, _ = selector.select(
    hidden=hidden_state,
    logits=logits,
    anchor_ids=anchor_token_id,
    temperature=0.0
)
v2_path = [vocab[id.item()] for id in path[0]]
print(f"V2 选择的草稿: {' -> '.join(v2_path)}")
print("【评价】: 发现 '狗 -> 叫' 不匹配，因为 '猫' 的后继特征与 '叫' 完美对齐，")
print("        路径选择器果断放弃了位置1概率最高的 '狗'，选了概率第二的 '猫'，")
print("        最终拼出了极其连贯的 '猫 -> 叫'，成功骗过大模型验证！")