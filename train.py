import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.base import GPTConfig
from datasets.Loader import SeqMonkeyDataset
from modules.GPT import GPT


def main():
    config = GPTConfig()
    dataset = SeqMonkeyDataset(config)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataset_loader, val_dataset_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True), \
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    gpt = GPT(config)
    gpt = gpt.to(config.device)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.t_step, eta_min=0
    )
    for epoch in range(config.epoch):
        train(epoch + 1, gpt, optimizer, scheduler, train_dataset_loader, val_dataset_loader, config)


def train(
        epoch: int,
        model: GPT,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: GPTConfig
):
    model.train()
    total_loss = 0
    loader_tqdm = tqdm(train_dataloader)
    for batch_idx, (x, y) in enumerate(loader_tqdm):
        x, y = x.to(config.device), y.to(config.device)
        # 正向与反向传播
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        # 学习率调整
        scheduler.step()
        total_loss += loss.item()
        if batch_idx % 5 == 0:
            loader_tqdm.set_description(
                f"epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item():.4f}, mean_loss: {total_loss / (batch_idx + 1):.4f}"
            )

    if epoch % 4 == 0:
        eval(epoch, model, val_dataloader, config)
    return total_loss


def eval(
        epoch: int,
        model: GPT,
        val_dataloader: DataLoader,
        config: GPTConfig
):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_dataloader):
            x, y = x.to(config.device), y.to(config.device)
            logits, loss = model(x, y)
            total_loss += loss.item()
        print(f"eval epoch: {epoch}, mean_loss: {total_loss / len(val_dataloader):.4f}")
    # 保存模型
    torch.save(model.state_dict(), os.path.join(config.checkpoint_path, f"gpt_epoch_{epoch}.pt"))
    with open(os.path.join(config.checkpoint_path, "loss.txt"), "a") as f:
        f.write(f"epoch: {epoch}, loss: {total_loss / len(val_dataloader):.4f}\n")
    return total_loss


if __name__ == '__main__':
    main()
