import os

import tiktoken
import torch
from tiktoken import Encoding

from configs.base import GPTConfig
from modules.GPT import GPT


def main():
    config = GPTConfig()

    gpt = GPT(config)
    gpt = gpt.to(config.device)
    gpt.eval()

    # 分布式训练下保存的模型键值对中包含module.
    state_dict = torch.load(os.path.join(config.checkpoint_path, "gpt_epoch_28.pt"), weights_only=True)
    keys = list(state_dict.keys())
    for key in keys:
        state_dict[key[7:]] = state_dict.pop(key)
    gpt.load_state_dict(state_dict)
    print("加载模型成功")

    # 加载分词器
    enc = tiktoken.get_encoding("gpt2")
    eos = enc.encode(
        "<|endoftext|>",
        allowed_special={
            "<|endoftext|>"
        }
    )[0]

    infer(
        "新年快乐。",
        eos, config.max_new_tokens, gpt, enc, config
    )


def infer(
        input_text: str,
        eos: int,
        max_new_tokens: int,
        model: GPT,
        tokenizer: Encoding,
        config: GPTConfig
):
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text)
        print("输入:", tokenizer.decode(input_ids))
        input_tensor = torch.tensor(input_ids[-config.block_size:]).unsqueeze(0).to(config.device)
        tokens = model.generate(input_tensor, max_new_tokens, eos, only_new_tokens=True)[0]
        print("输出：", tokenizer.decode(tokens.tolist()))


if __name__ == '__main__':
    main()
