import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from configs.base import GPTConfig
from modules.Attention import MultiHeadAttention, FeedForward


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # (token embedding, position embedding, norm, mlp)
        # <todo> position embedding 从 固定 embedding | 可学习 embedding 变为 rope
        # <todo> norm 从 layer norm 变为 RMSNorm
        # <todo> mlp 从 feed forward 变为 swiglu
        # <todo> mha 从 multi head attention 变为 gqa

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 可学习的位置编码
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[GPTBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_final = nn.LayerNorm(config.n_embd)
        # bias为False
        # 因为 x + b 在 softmax 中没有意义
        # 加和不加结果是一样的
        self.output_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 现在的small-language-model 会使用 tie_weight 来共享embedding和输出层的权重
        # 非常关键的知识点
        # 如果一个Linear为(a, b)那么它的weight为(b, a)
        self.output_proj.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        else:
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        # idx [batch_size, seq_len] 输入
        # targets [batch_size, seq_len] 目标
        batch_size, seq_len = idx.size()
        token_embedded = self.token_embedding(idx)  # [batch_size, seq_len, n_embd]
        pos_emb = self.position_embedding(
            torch.arange(seq_len, device=idx.device)
        )  # [seq_len, n_embd]

        x = token_embedded + pos_emb
        x = self.blocks(x)  # [batch_size, seq_len, n_embd]
        x = self.ln_final(x)  # [batch_size, seq_len, n_embd]
        logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
            # 重整形状
            logits = logits.view(batch, seq_len, vocab_size)
        return logits, loss

    def generate(self, ids, max_new_tokens, eos: int, only_new_tokens: bool = True):
        # ids [batch_size, seq_len]
        assert ids.size(0) == 1, "only supports batch_size=1"
        batch_size, seq_len = ids.size()
        bar = tqdm(range(max_new_tokens), desc="推理生成中")
        while True:
            ids_cond = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
            logits, _ = self(ids_cond)  # [batch_size, seq_len, vocab_size]
            # 取最后一个token的概率分布
            logits = logits[:, -1, :]
            # 使用top-k采样
            top_k_logits = torch.topk(logits, self.config.top_k, dim=-1)[0]
            logits = logits.masked_fill(logits < top_k_logits[:, -1].unsqueeze(-1), -float("inf"))
            probs = F.softmax(logits, dim=-1)
            # 采样
            new_tokens = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, new_tokens], dim=-1)
            # 更新进度条
            bar.update(1)
            if new_tokens[0].item() == eos or ids.size(1) - seq_len >= max_new_tokens:
                bar.set_description("推理生成完成")
                break
        if only_new_tokens:
            return ids[:, seq_len:]
        return ids
