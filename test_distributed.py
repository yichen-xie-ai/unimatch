#!/usr/bin/env python3
"""
测试分布式训练设置的诊断脚本
运行方式: torchrun --nproc_per_node=2 test_distributed.py
"""

import os
import torch
import torch.distributed as dist

def main():
    # 打印环境变量
    print(f"Process PID: {os.getpid()}")
    print(f"RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    
    # 初始化进程组
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"\n[Rank {local_rank}] 开始初始化分布式进程组...")
    
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] 分布式初始化成功! World size: {world_size}")
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    print(f"[Rank {rank}] 使用设备: {device}")
    
    # 测试同步
    print(f"[Rank {rank}] 测试 barrier 同步...")
    dist.barrier()
    print(f"[Rank {rank}] Barrier 同步成功!")
    
    # 创建简单模型
    print(f"[Rank {rank}] 创建测试模型...")
    model = torch.nn.Linear(10, 10).to(device)
    
    # 测试 DDP
    print(f"[Rank {rank}] 初始化 DDP...")
    dist.barrier()
    
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    
    print(f"[Rank {rank}] DDP 初始化成功!")
    
    # 测试前向传播
    print(f"[Rank {rank}] 测试前向传播...")
    x = torch.randn(2, 10).to(device)
    y = model(x)
    print(f"[Rank {rank}] 前向传播成功! 输出形状: {y.shape}")
    
    # 测试反向传播
    print(f"[Rank {rank}] 测试反向传播...")
    loss = y.sum()
    loss.backward()
    print(f"[Rank {rank}] 反向传播成功!")
    
    # 最终同步
    dist.barrier()
    print(f"[Rank {rank}] 所有测试通过! ✓")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

