#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE数据增强测试脚本
使用生成的train数据进行数据增强测试
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径，以便导入CVAE模块
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from CVAE_augmentation_customized import augment_data_with_cvae


def main():
    # ====== Configuration ======
    # 使用刚才生成的train数据
    data_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS_local/RNA-seq_augmentation/test/SKCMPositive_5-2_train.csv"
    
    # 输出路径和文件名
    output_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS_local/RNA-seq_augmentation/test"
    output_filename = "SKCMPositive_5-2_train_CVAE_augmented"
    
    # 数据配置
    batch_col = "groups"  # 使用groups列作为batch列
    samples_per_batch = 500  # 每个batch生成的样本数
    
    # 训练超参数（可以根据需要调整）
    kl_weight = 0.05
    num_epochs = 1000
    learning_rate = 1e-3
    warmup_epochs = 50
    batch_size = 64
    early_stop = True
    early_stop_patience = 200
    
    # 模型参数
    latent_dim = 32
    dropout = 0.2
    
    # 其他参数
    device = "cpu"  # 如果有GPU可以改为 "cuda"
    log_transform = True
    
    print("=" * 60)
    print("CVAE Data Augmentation Test")
    print("=" * 60)
    print(f"Input data: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Output filename: {output_filename}")
    print(f"Batch column: {batch_col}")
    print(f"Samples per batch: {samples_per_batch}")
    print(f"KL weight: {kl_weight}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    # ====== 执行数据增强 ======
    augmented_data = augment_data_with_cvae(
        data=data_path,
        output_path=output_path,
        output_filename=output_filename,
        batch_col=batch_col,
        samples_per_batch=samples_per_batch,
        kl_weight=kl_weight,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
        batch_size=batch_size,
        early_stop=early_stop,
        early_stop_patience=early_stop_patience,
        latent_dim=latent_dim,
        dropout=dropout,
        log_transform=log_transform,
        device=device,
        save_model_path=None,  # 不保存模型
        save_training_log=True,  # 保存训练日志
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"Generated {len(augmented_data)} augmented samples")
    print("=" * 60)
    
    return augmented_data


if __name__ == "__main__":
    augmented_data = main()
