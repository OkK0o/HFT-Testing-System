#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本，用于验证修改后的compute_factor函数是否正确工作
"""

import pandas as pd
import numpy as np
from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from factor_compute import compute_factor, compute_multiple_factors
import gc
import sys

def test_single_factor(df, factor_name):
    """
    测试单个因子的计算和返回
    """
    print(f"\n===== 测试计算因子: {factor_name} =====")
    
    # 检查因子是否已存在
    if factor_name in df.columns:
        print(f"因子 {factor_name} 已存在于数据中，将先删除")
        df = df.drop(columns=[factor_name])
    
    # 记录原始列
    original_cols = df.columns.tolist()
    print(f"原始数据形状: {df.shape}")
    print(f"原始列数: {len(original_cols)}")
    
    # 计算因子
    print(f"\n计算因子...")
    result_df = compute_factor(df, factor_name)
    
    # 验证结果
    new_cols = result_df.columns.tolist()
    print(f"计算后数据形状: {result_df.shape}")
    print(f"计算后列数: {len(new_cols)}")
    
    # 检查因子是否被成功添加
    if factor_name in result_df.columns:
        print(f"\n✓ 成功! 因子 {factor_name} 已添加到结果DataFrame")
        print(f"因子 {factor_name} 的前10个值:")
        print(result_df[factor_name].head(10))
        
        # 检查非空值
        non_null_count = result_df[factor_name].count()
        print(f"因子 {factor_name} 非空值数量: {non_null_count}/{len(result_df)} ({non_null_count/len(result_df)*100:.2f}%)")
        
        return True
    else:
        print(f"\n✗ 失败! 因子 {factor_name} 未添加到结果DataFrame")
        added_cols = set(new_cols) - set(original_cols)
        if added_cols:
            print(f"不过添加了其他列: {added_cols}")
        return False

def test_multiple_factors(df, factor_names):
    """
    测试多个因子的批量计算
    """
    print(f"\n===== 测试批量计算 {len(factor_names)} 个因子 =====")
    
    # 删除已存在的因子
    existing_factors = [f for f in factor_names if f in df.columns]
    if existing_factors:
        print(f"以下因子已存在于数据中，将先删除: {existing_factors}")
        df = df.drop(columns=existing_factors)
    
    # 记录原始列
    original_cols = df.columns.tolist()
    print(f"原始数据形状: {df.shape}")
    print(f"原始列数: {len(original_cols)}")
    
    # 批量计算因子
    print(f"\n批量计算因子...")
    result_df = compute_multiple_factors(df, factor_names, adjust_sign=False)
    
    # 验证结果
    new_cols = result_df.columns.tolist()
    print(f"计算后数据形状: {result_df.shape}")
    print(f"计算后列数: {len(new_cols)}")
    
    # 检查每个因子是否被成功添加
    success_count = 0
    for factor in factor_names:
        if factor in result_df.columns:
            success_count += 1
            print(f"✓ 因子 {factor} 成功添加")
        else:
            print(f"✗ 因子 {factor} 添加失败")
    
    print(f"\n成功率: {success_count}/{len(factor_names)} ({success_count/len(factor_names)*100:.2f}%)")
    
    return success_count == len(factor_names)

def main():
    # 加载测试数据
    print("加载测试数据...")
    try:
        df = pd.read_feather("df_with_smooth.feather")
        print(f"成功加载数据，形状: {df.shape}")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        sys.exit(1)
    
    # 注册所有因子
    register_all_factors()
    print("已注册所有因子")
    
    # 获取所有可用因子
    all_factors = FactorManager.get_factor_names()
    print(f"可用因子总数: {len(all_factors)}")
    
    # 测试momentum_300因子
    test_single_factor(df, "momentum_300")
    
    # 测试一些其他因子
    test_factors = ["realized_vol_100", "order_book_imbalance", "minute_rsi_120"]
    test_multiple_factors(df, test_factors)
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    main() 