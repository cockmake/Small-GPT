import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from configs.base import GPTConfig
from datasets.Loader import SeqMonkeyDataset
from modules.GPT import GPT


def main(rank, world_size):
    # 初始化分布式环境
    # linux 下使用 nccl
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    # windows 下使用 gloo
    # dist.init_process_group(backend="gloo", init_method="env://?use_libuv=False", world_size=world_size, rank=rank)
    torch.manual_seed(723)  # 保证各进程的随机数一致性

    # 配置和数据集加载
    config = GPTConfig()
    device = torch.device(f"cuda:{rank}")
    dataset = SeqMonkeyDataset(config)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    # 使用 DistributedSampler 分割数据集
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)

    # 初始化模型并分布式包装
    model = GPT(config).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.t_step, eta_min=0)

    # 开始训练
    for epoch in range(config.epoch):
        train_sampler.set_epoch(epoch)  # 确保每个 epoch 数据分布不同
        train(epoch + 1, model, optimizer, scheduler, train_loader, val_loader, config, rank)

    # 销毁分布式环境
    dist.destroy_process_group()


def train(epoch, model, optimizer, scheduler, train_dataloader, val_dataloader, config, rank):
    model.train()
    total_loss = 0
    # 主进程显示进度条
    loader_tqdm = tqdm(train_dataloader) if rank == 0 else train_dataloader
    for batch_idx, (x, y) in enumerate(loader_tqdm):
        x, y = x.to(rank), y.to(rank)
        optimizer.zero_grad()  # 清空梯度
        logits, loss = model(x, y)  # 正向传播
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        scheduler.step()  # 调整学习率

        total_loss += loss.item()
        if rank == 0 and batch_idx % 5 == 0:
            loader_tqdm.set_description(
                f"epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item():.4f}, mean_loss: {total_loss / (batch_idx + 1):.4f}"
            )

    # 每隔 4 个 epoch 验证
    if epoch % 4 == 0 and rank == 0:
        eval(epoch, model, val_dataloader, config, rank)
    return total_loss


def eval(epoch, model, val_dataloader, config, rank):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_dataloader):
            x, y = x.to(rank), y.to(rank)
            logits, loss = model(x, y)  # 验证模型
            total_loss += loss.item()
        if rank == 0:
            print(f"eval epoch: {epoch}, mean_loss: {total_loss / len(val_dataloader):.4f}")

    # 主进程保存模型和记录日志
    if rank == 0:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.checkpoint_path, f"gpt_epoch_{epoch}.pt"))
        with open(os.path.join(config.checkpoint_path, "loss.txt"), "a") as f:
            f.write(f"epoch: {epoch}, loss: {total_loss / len(val_dataloader):.4f}\n")
    return total_loss


if __name__ == '__main__':
    # 获取 GPU 数量并设置多进程训练
    world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 设置主节点地址
    os.environ["MASTER_PORT"] = "29500"  # 设置通信端口

    # 使用多进程启动训练
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
