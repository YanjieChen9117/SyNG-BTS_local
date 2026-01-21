#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量数据增强脚本
对5个cancer type、3种normalization方法、20个batch进行全流程数据增强
"""

import sys
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# 添加项目路径以便导入模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "SyNG-BTS" / "python"))
sys.path.insert(0, str(project_root / "SyNG-BTS" / "vignettes"))

from data_split import split_data
from CVAE_augmentation_customized import augment_data_with_cvae


# ========== 配置参数 ==========
CANCER_TYPES = ["COAD", "LAML", "PAAD", "READ", "SKCM"]
NORMALIZATION_METHODS = ["raw", "TC", "DESeq"]
NUM_BATCHES = 20

# 数据根目录
DATA_ROOT = project_root / "data"
OUTPUT_ROOT = project_root / "RNA-seq_augmentation_data" / "main"

# 状态文件路径
STATUS_FILE = project_root / "scripts" / "RNA-seq_augmentation_status.csv"

# CVAE参数（使用CVAE_customized_test.py中的参数）
CVAE_CONFIG = {
    "batch_col": "groups",
    "samples_per_batch": 500,
    "kl_weight": 0.05,
    "num_epochs": 1000,
    "learning_rate": 1e-3,
    "warmup_epochs": 50,
    "batch_size": 64,
    "early_stop": True,
    "early_stop_patience": 200,
    "latent_dim": 32,
    "dropout": 0.2,
    "log_transform": True,
    "device": "cpu",
    "save_model_path": None,
    "save_training_log": True,
    "verbose": True,
}


def load_status_file() -> pd.DataFrame:
    """加载状态文件，如果不存在则创建新的"""
    if STATUS_FILE.exists():
        df = pd.read_csv(STATUS_FILE)
    else:
        df = pd.DataFrame(columns=[
            "cancer_type", "normalization", "batch_num", 
            "status", "start_time", "end_time", "duration_seconds", "error_message"
        ])
    return df


def save_status_file(df: pd.DataFrame):
    """保存状态文件"""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATUS_FILE, index=False)


def get_data_filename(cancer_type: str, normalization: str) -> str:
    """获取数据文件名（不含路径，不含扩展名）"""
    if normalization == "raw":
        return f"{cancer_type}Positive_5-2"
    else:
        return f"{cancer_type}Positive_5-2_{normalization}"


def get_data_path(cancer_type: str, normalization: str) -> Path:
    """获取输入数据路径"""
    data_dir = DATA_ROOT / f"{cancer_type}_5-2_with_group"
    filename_base = get_data_filename(cancer_type, normalization)
    filename = f"{filename_base}.csv"
    return data_dir / filename


def get_output_path(cancer_type: str, normalization: str, batch_num: int) -> Path:
    """获取输出路径"""
    return OUTPUT_ROOT / f"{cancer_type}_5-2" / normalization / f"batch_{batch_num}"


def get_random_seed(batch_num: int) -> int:
    """根据batch编号生成不同的random seed"""
    # 使用一个基础seed加上batch编号，确保每个batch的seed都不同
    base_seed = 42
    return base_seed + batch_num * 1000


def check_task_completed(df: pd.DataFrame, cancer_type: str, 
                        normalization: str, batch_num: int) -> bool:
    """检查任务是否已完成"""
    mask = (
        (df["cancer_type"] == cancer_type) &
        (df["normalization"] == normalization) &
        (df["batch_num"] == batch_num) &
        (df["status"] == "success")
    )
    return mask.any()


def update_status(df: pd.DataFrame, cancer_type: str, normalization: str,
                 batch_num: int, status: str, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None, duration: Optional[float] = None,
                 error_message: Optional[str] = None) -> pd.DataFrame:
    """更新状态文件"""
    # 查找是否已存在记录
    mask = (
        (df["cancer_type"] == cancer_type) &
        (df["normalization"] == normalization) &
        (df["batch_num"] == batch_num)
    )
    
    new_row = {
        "cancer_type": cancer_type,
        "normalization": normalization,
        "batch_num": batch_num,
        "status": status,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
        "duration_seconds": duration,
        "error_message": error_message,
    }
    
    if mask.any():
        # 更新现有记录
        idx = df[mask].index[0]
        for key, value in new_row.items():
            df.at[idx, key] = value
    else:
        # 添加新记录
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df


def process_single_task(cancer_type: str, normalization: str, batch_num: int) -> Tuple[bool, Optional[str]]:
    """
    处理单个任务（一个cancer type + normalization + batch的组合）
    
    返回: (success: bool, error_message: Optional[str])
    """
    try:
        print(f"\n{'='*80}")
        print(f"处理任务: {cancer_type} - {normalization} - batch_{batch_num}")
        print(f"{'='*80}")
        
        # 1. 获取数据路径
        data_path = get_data_path(cancer_type, normalization)
        if not data_path.exists():
            error_msg = f"数据文件不存在: {data_path}"
            print(f"错误: {error_msg}")
            return False, error_msg
        
        # 2. 获取输出路径
        output_path = get_output_path(cancer_type, normalization, batch_num)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 3. 数据分割
        print(f"\n步骤1: 数据分割")
        print(f"输入数据: {data_path}")
        print(f"输出路径: {output_path}")
        
        random_seed = get_random_seed(batch_num)
        print(f"使用random seed: {random_seed}")
        
        # 使用与输入文件相同的命名规则
        output_filename = get_data_filename(cancer_type, normalization)
        train_data, test_data = split_data(
            data=str(data_path),
            output_path=str(output_path),
            output_filename=output_filename,
            random_seed=random_seed,
            verbose=True
        )
        
        # 4. 数据增强（对train数据）
        print(f"\n步骤2: CVAE数据增强")
        train_file = output_path / f"{output_filename}_train.csv"
        
        if not train_file.exists():
            error_msg = f"训练数据文件不存在: {train_file}"
            print(f"错误: {error_msg}")
            return False, error_msg
        
        augmented_filename = f"{output_filename}_train_CVAE_augmented"
        
        print(f"输入训练数据: {train_file}")
        print(f"输出文件名: {augmented_filename}")
        print(f"CVAE配置: {CVAE_CONFIG}")
        
        augmented_data = augment_data_with_cvae(
            data=str(train_file),
            output_path=str(output_path),
            output_filename=augmented_filename,
            **CVAE_CONFIG
        )
        
        print(f"\n✓ 任务完成: {cancer_type} - {normalization} - batch_{batch_num}")
        print(f"  生成了 {len(augmented_data)} 个增强样本")
        
        return True, None
        
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(f"\n✗ 错误: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg


def main():
    """主函数"""
    print("="*80)
    print("批量数据增强脚本")
    print("="*80)
    print(f"Cancer types: {CANCER_TYPES}")
    print(f"Normalization methods: {NORMALIZATION_METHODS}")
    print(f"Number of batches: {NUM_BATCHES}")
    print(f"Total tasks: {len(CANCER_TYPES) * len(NORMALIZATION_METHODS) * NUM_BATCHES}")
    print(f"Status file: {STATUS_FILE}")
    print("="*80)
    
    # 加载状态文件
    status_df = load_status_file()
    print(f"\n已加载状态文件，当前有 {len(status_df)} 条记录")
    
    # 统计需要处理的任务
    total_tasks = 0
    skipped_tasks = 0
    for cancer_type in CANCER_TYPES:
        for normalization in NORMALIZATION_METHODS:
            for batch_num in range(1, NUM_BATCHES + 1):
                total_tasks += 1
                if check_task_completed(status_df, cancer_type, normalization, batch_num):
                    skipped_tasks += 1
    
    print(f"总任务数: {total_tasks}")
    print(f"已跳过任务数: {skipped_tasks}")
    print(f"待处理任务数: {total_tasks - skipped_tasks}")
    
    # 处理所有任务
    processed = 0
    success_count = 0
    failed_count = 0
    
    for cancer_type in CANCER_TYPES:
        for normalization in NORMALIZATION_METHODS:
            for batch_num in range(1, NUM_BATCHES + 1):
                # 检查是否已完成
                if check_task_completed(status_df, cancer_type, normalization, batch_num):
                    print(f"\n跳过已完成任务: {cancer_type} - {normalization} - batch_{batch_num}")
                    continue
                
                # 记录开始时间
                start_time = datetime.now()
                status_df = update_status(
                    status_df, cancer_type, normalization, batch_num,
                    status="running", start_time=start_time
                )
                save_status_file(status_df)
                
                # 处理任务
                success, error_message = process_single_task(
                    cancer_type, normalization, batch_num
                )
                
                # 记录结束时间
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # 更新状态
                status = "success" if success else "failed"
                status_df = update_status(
                    status_df, cancer_type, normalization, batch_num,
                    status=status, start_time=start_time, end_time=end_time,
                    duration=duration, error_message=error_message
                )
                save_status_file(status_df)
                
                processed += 1
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                
                print(f"\n进度: {processed}/{total_tasks - skipped_tasks} 已完成")
                print(f"成功: {success_count}, 失败: {failed_count}")
    
    # 最终统计
    print("\n" + "="*80)
    print("批量处理完成！")
    print("="*80)
    print(f"总任务数: {total_tasks}")
    print(f"已跳过: {skipped_tasks}")
    print(f"本次处理: {processed}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"状态文件: {STATUS_FILE}")
    print("="*80)


if __name__ == "__main__":
    main()
