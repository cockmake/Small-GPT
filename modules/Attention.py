import torch
from torch import nn

from configs.base import GPTConfig


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.head_size = config.head_size

        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)

        # attention mask
        # 使用register_buffer注册一个attention_mask的buffer
        self.register_buffer(
            "attention_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size),
                diagonal=0
            )
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, "Input tensor shape should be [batch_size, seq_len, hidden_dim]"
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)  # [batch_size, seq_len, head_size]
        q = self.query(x)
        v = self.value(x)

        # 注意要除以head_size的平方根
        # 就是概率分布，样本均值的分布为N~(a, b/ n ** 0.5)
        # 为了使得梯度更加稳定
        weight = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)  # [batch_size, seq_len, seq_len]
        weight = weight.masked_fill(
            # 有可能输入矩阵的长度小于block_size
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )

        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return weight, out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.heads = nn.ModuleList([
            SingleHeadAttention(config) for _ in range(config.n_head)
        ])

        # 在这里n_head * head_size = hidden_dim
        self.proj = nn.Linear(config.n_head * config.head_size, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([
            head(x)[1] for head in self.heads
        ], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.mlp(x)


def main():
    t = torch.randn(2, 512, 768)
    config = GPTConfig()
    model = MultiHeadAttention(config)
    out = model(t)
    print(out.size())


if __name__ == '__main__':
    main()
