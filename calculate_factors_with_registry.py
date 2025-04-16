import pandas as pd
import numpy as np
import os
from typing import List, Dict
from pathlib import Path
import gc
import time

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors, register_smoothed_factors
from factor_compute import compute_multiple_factors, adjust_factor_signs, smooth_factors
from factors_test import FactorsTester

# 设置文件路径
input_file = 'processed_volume_data.feather'  # 使用处理好的数据
output_dir = 'factors_results'
output_file = os.path.join(output_dir, 'data_with_factors.feather')

# 保存批次大小
BATCH_SIZE = 10

# 如果输出目录不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误：输入文件 {input_file} 不存在，请先运行数据处理脚本")
    exit(1)

# 定义一个函数来保存当前的计算结果
def save_current_results(df, computed_factors):
    """保存当前计算结果到文件"""
    print(f"\n保存当前结果到: {output_file}")
    start_time = time.time()
    df.to_feather(output_file)
    print(f"数据保存成功，耗时: {time.time() - start_time:.2f}秒")
    print(f"已计算的因子数量: {len(computed_factors)}")

# 定义一个函数来获取当前需要计算的因子
def get_factors_to_compute(all_factors, df):
    """获取需要计算的因子列表"""
    existing_factors = set(df.columns)
    registered_factors = set(all_factors['name'])
    already_computed = existing_factors.intersection(registered_factors)
    to_compute = registered_factors - already_computed
    
    print(f"\n已经计算的因子数量: {len(already_computed)}")
    print(f"还需计算的因子数量: {len(to_compute)}")
    
    return list(to_compute)

print(f"开始处理文件: {input_file}")

# 读取处理好的数据
print("读取处理好的数据...")
df = pd.read_feather(input_file)
print(f"数据形状: {df.shape}")

# 注册所有因子
print("注册所有因子...")
register_all_factors()

# 获取所有因子信息
factor_info = FactorManager.get_factor_info()
print(f"共注册了 {len(factor_info)} 个因子")

# 计算所有因子
# 我们按类别分批计算，避免内存问题
print("\n开始按类别计算因子...")

# 定义要计算的因子类别顺序
category_order = [
    'momentum', 'volatility', 'volume', 'orderbook', 
    'microstructure', 'time', 'term_structure', 'composite'
]

result_df = df.copy()
all_computed_factors = []
factor_counter = 0

# 如果输出文件已存在，先读取它
if os.path.exists(output_file):
    print(f"发现已有输出文件: {output_file}，读取现有数据...")
    try:
        result_df = pd.read_feather(output_file)
        print(f"读取成功，数据形状: {result_df.shape}")
    except Exception as e:
        print(f"读取已有文件出错: {str(e)}，将使用原始数据")
        result_df = df.copy()

# 获取需要计算的因子
factors_to_compute = get_factors_to_compute(factor_info, result_df)

# 将因子按类别分组
factors_by_category = {}
for idx, row in factor_info.iterrows():
    if row['name'] in factors_to_compute:
        if row['category'] not in factors_by_category:
            factors_by_category[row['category']] = []
        factors_by_category[row['category']].append(row['name'])

for category in category_order:
    # 获取当前类别的因子
    current_factors = factors_by_category.get(category, [])
    if not current_factors:
        print(f"\n类别 '{category}' 没有待计算因子，跳过")
        continue
    
    print(f"\n计算 {category} 类因子 (共 {len(current_factors)} 个)...")
    
    # 将当前类别的因子分成小批次计算
    for i in range(0, len(current_factors), BATCH_SIZE):
        batch_factors = current_factors[i:i+BATCH_SIZE]
        print(f"\n计算批次 {i//BATCH_SIZE + 1}/{(len(current_factors)+BATCH_SIZE-1)//BATCH_SIZE}: {batch_factors}")
        
        # 使用compute_multiple_factors计算小批次因子
        try:
            result_df = compute_multiple_factors(
                df=result_df, 
                factor_names=batch_factors,
                adjust_sign=False,  # 先不调整符号
                use_period=120  # 使用120periods的return
            )
            
            # 检查哪些因子成功计算
            computed_batch = [f for f in batch_factors if f in result_df.columns]
            all_computed_factors.extend(computed_batch)
            factor_counter += len(computed_batch)
            
            print(f"成功计算 {len(computed_batch)}/{len(batch_factors)} 个因子")
            
            # 每计算完一个批次，保存当前结果
            save_current_results(result_df, all_computed_factors)
            
            # 重新读取文件，确保数据一致性并释放内存
            result_df = pd.read_feather(output_file)
            
            # 更新待计算因子列表
            factors_to_compute = get_factors_to_compute(factor_info, result_df)
            
            # 每个批次计算完后清理内存
            gc.collect()
        except Exception as e:
            print(f"计算批次 {i//BATCH_SIZE + 1} 时出错: {str(e)}")
            # 保存当前进度
            save_current_results(result_df, all_computed_factors)
            # 重新读取文件
            result_df = pd.read_feather(output_file)
            # 更新待计算因子列表
            factors_to_compute = get_factors_to_compute(factor_info, result_df)
            continue

print(f"\n所有类别因子计算完成！共计算 {len(all_computed_factors)} 个因子")

# 计算因子平滑（如果有需要）
print("\n开始计算因子平滑...")
try:
    # 只选择几个重要的微观结构和流动性因子进行平滑
    factors_to_smooth = [
        'order_book_imbalance',      # 订单簿不平衡
        'effective_spread',          # 有效价差
        'amihud_illiquidity',        # 非流动性因子
        'order_flow_toxicity',       # 订单流毒性
        'price_impact',              # 价格冲击因子
        'hft_trend',                 # 高频趋势
        'microstructure_momentum',   # 微观结构动量
        'intraday_seasonality'       # 日内季节性
    ]
    
    # 只保留实际计算成功的因子
    factors_to_smooth = [f for f in factors_to_smooth if f in all_computed_factors]
    
    print(f"选择 {len(factors_to_smooth)} 个重要因子进行平滑")
    print("平滑的因子列表:")
    for f in factors_to_smooth:
        print(f"  - {f}")
    
    # 应用EMA平滑，使用不同窗口
    ema_periods = [600, 1200]  # 10分钟、20分钟
    
    for period in ema_periods:
        print(f"\n应用 EMA{period} 平滑...")
        result_df, smoothed_names = smooth_factors(
            df=result_df,
            factor_names=factors_to_smooth,
            windows=period,
            methods='ema',
            register_factors=True
        )
        all_computed_factors.extend(smoothed_names)
        
        # 保存当前结果
        save_current_results(result_df, all_computed_factors)
        
        # 重新读取文件
        result_df = pd.read_feather(output_file)
    
    print(f"因子平滑完成，共生成 {len(factors_to_smooth) * len(ema_periods)} 个平滑因子")
except Exception as e:
    print(f"计算因子平滑时出错: {str(e)}")
    # 保存当前结果
    save_current_results(result_df, all_computed_factors)

# 输出统计信息
print("\n数据统计信息:")
print(f"- 原始数据形状: {df.shape}")
print(f"- 最终数据形状: {result_df.shape}")
print(f"- 计算的因子数量: {len(all_computed_factors)}")

print("\n处理完成！") 