#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据split函数
用于将数据分割为train和test集，供后续脚本调用
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Union, Optional, Tuple


def split_data(
    data: Union[str, pd.DataFrame],
    output_path: str,
    output_filename: str,
    random_seed: int = 42,
    verbose: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数据split函数
    
    参数:
        data: 需要被处理的数据，可以是文件路径（str）或DataFrame
        output_path: 输出路径
        output_filename: 输出文件名（不含扩展名）
        random_seed: 随机种子，用于shuffle数据
        verbose: 是否打印详细信息
        **kwargs: 其他可选参数（如传递给pd.read_csv的参数）
    
    返回:
        tuple: (train_data, test_data) DataFrame元组
    """
    # 1. 读取数据
    if isinstance(data, str):
        if verbose:
            print(f"正在读取数据文件: {data}")
        df = pd.read_csv(data, **kwargs)
    elif isinstance(data, pd.DataFrame):
        if verbose:
            print("使用提供的DataFrame")
        df = data.copy()
    else:
        raise TypeError("data参数必须是文件路径（str）或DataFrame")
    
    # 2. 检查列
    has_samples = 'samples' in df.columns
    has_groups = 'groups' in df.columns
    
    if verbose:
        print(f"\n列检查结果:")
        print(f"  包含 'samples' 列: {has_samples}")
        print(f"  包含 'groups' 列: {has_groups}")
    
    # 3. 计算数据size
    # 排除samples和groups列，计算marker数量
    marker_cols = [col for col in df.columns if col not in ['samples', 'groups']]
    num_markers = len(marker_cols)
    num_samples = len(df)
    
    if verbose:
        print(f"\n数据size:")
        print(f"  Marker数量（非samples和groups的列数）: {num_markers}")
        print(f"  Sample数量（行数）: {num_samples}")
    
    # 4. 根据sample数量确定split比例
    if num_samples > 200:
        train_ratio = 0.8
        test_ratio = 0.2
        split_strategy = "80% train, 20% test"
    elif num_samples > 100:
        train_ratio = 0.5
        test_ratio = 0.5
        split_strategy = "50% train, 50% test"
    else:
        train_ratio = 1.0
        test_ratio = 1.0
        split_strategy = "使用原始数据作为train和test（全部数据）"
    
    if verbose:
        print(f"\nSplit策略: {split_strategy}")
        print(f"  Sample数量: {num_samples}")
    
    # 5. 执行split
    # 设置随机种子
    np.random.seed(random_seed)
    
    if train_ratio == 1.0 and test_ratio == 1.0:
        # 如果sample数量 <= 100，train和test都使用全部数据
        train_data = df.copy()
        test_data = df.copy()
        if verbose:
            print(f"\n由于sample数量 <= 100，train和test都使用全部 {num_samples} 个样本")
    else:
        # Shuffle数据
        df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # 计算split点
        train_size = int(num_samples * train_ratio)
        
        # 分割数据
        train_data = df_shuffled.iloc[:train_size].copy()
        test_data = df_shuffled.iloc[train_size:].copy()
        
        if verbose:
            print(f"\nSplit结果:")
            print(f"  Train集: {len(train_data)} 个样本 ({len(train_data)/num_samples*100:.1f}%)")
            print(f"  Test集: {len(test_data)} 个样本 ({len(test_data)/num_samples*100:.1f}%)")
    
    # 6. 确保输出路径存在
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 7. 保存文件
    train_filename = f"{output_filename}_train.csv"
    test_filename = f"{output_filename}_test.csv"
    
    train_filepath = output_path / train_filename
    test_filepath = output_path / test_filename
    
    train_data.to_csv(train_filepath, index=False)
    test_data.to_csv(test_filepath, index=False)
    
    if verbose:
        print(f"\n文件已保存:")
        print(f"  Train文件: {train_filepath}")
        print(f"  Test文件: {test_filepath}")
    
    return train_data, test_data


if __name__ == "__main__":
    # 测试代码
    test_data_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS_local/data/SKCM_5-2_with_group/SKCMPositive_5-2.csv"
    test_output_path = "/Users/yanjiechen/Documents/Github/SyNG-BTS_local/RNA-seq_augmentation/test"
    test_output_filename = "SKCMPositive_5-2"
    test_random_seed = 42
    
    print("=" * 60)
    print("数据Split测试")
    print("=" * 60)
    
    train_df, test_df = split_data(
        data=test_data_path,
        output_path=test_output_path,
        output_filename=test_output_filename,
        random_seed=test_random_seed,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
