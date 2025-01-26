import json

import tiktoken
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from configs.base import GPTConfig


class SeqMonkeyDataset(Dataset):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.enc = tiktoken.get_encoding("gpt2")

        self.block_size = config.block_size

        self.encoded_data = []
        # 特殊符号分割不同的文本
        # <|endoftext|> 为文本结束符号
        # eos 是个 list
        self.eos = self.enc.encode(
            "<|endoftext|>",
            allowed_special={
                "<|endoftext|>"
            }
        )

        raw_data = []
        with open(config.dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= config.dataset_total_train:
                    break
                try:
                    raw_data.append(json.loads(line.strip())["text"])
                except Exception as e:
                    continue

        flatten_encoded_data = []
        for text in tqdm(raw_data, desc="数据集编码中"):
            encoded_text = self.enc.encode(text)
            flatten_encoded_data.extend(encoded_text + self.eos)

        for i in range(0, len(flatten_encoded_data), self.block_size):
            # 要有target所以要+1
            # 每一行的长度为 block_size + 1
            # 这里是513
            chunk = flatten_encoded_data[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + self.eos * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
