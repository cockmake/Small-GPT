from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    epoch: int = 1
    t_step: int = 1000
    lr: float = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # 输入序列的最大长度
    block_size: int = 512
    batch_size: int = 4
    n_layer: int = 16
    n_head: int = 12
    # 与词表映射的embd_size的大小相同
    # 相同可以tie_embedding_weight
    n_embd: int = 768
    hidden_dim = n_embd
    dropout: float = 0.1
    head_size: int = n_embd // n_head
    # 词表大小位gpt2的词表大小
    vocab_size: int = 50257

    dataset_path = "./data/seqmonkey.jsonl"
    dataset_total_train = 2000
    checkpoint_path = "./checkpoints"
    # infer
    top_k = 5
    max_new_tokens = 512


