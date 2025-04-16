import pandas as pd
import numpy as np
import os
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from factors_test import FactorsTester, FactorTestConfig

# 设置文件路径
input_file = 'factors_results/data_with_all_factors.feather'
output_dir = 'factors_ic_results'
factor_list_file = 'factors_results/computed_factors_list.csv'

# 定义收益率周期
RETURN_PERIODS = [1, 5, 10, 20, 120]  # 1分钟、5分钟、10分钟、20分钟、120分钟

# 如果输出目录不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误：输入文件 {input_file} 不存在，请先运行因子计算脚本")
    exit(1)

print(f"开始处理文件: {input_file}")

# 读取包含所有因子的数据
print("读取包含因子的数据...")
df = pd.read_feather(input_file)
print(f"数据形状: {df.shape}")

# 读取因子列表
factor_names = []
if os.path.exists(factor_list_file):
    factor_df = pd.read_csv(factor_list_file)
    factor_names = factor_df['factor_name'].tolist()
    print(f"从文件加载了 {len(factor_names)} 个因子")
else:
    # 如果因子列表文件不存在，尝试从数据中推断
    print("因子列表文件不存在，尝试从数据中推断因子...")
    # 尝试识别因子列
    potential_factors = [col for col in df.columns if 
                        any(col.startswith(prefix) for prefix in 
                            ['momentum_', 'weighted_momentum_', 'realized_vol_', 'high_low_vol_',
                             'volume_intensity_', 'order_book_', 'effective_', 'amihud_',
                             'order_flow_', 'volume_synchronized_', 'bid_ask_', 'price_impact_',
                             'quote_slope_', 'price_reversal_', 'hft_trend', 'microstructure_',
                             'intraday_', 'term_premium', 'volume_price_', 'liquidity_adjusted_',
                             'minute_'])]
    factor_names = potential_factors
    print(f"从数据中推断出 {len(factor_names)} 个因子")

# 检查是否有收益率列
for period in RETURN_PERIODS:
    period_col = f'{period}period_return'
    if period_col not in df.columns:
        print(f"计算{period}期收益率...")
        df = FactorsTester.calculate_forward_returns(
            df=df, 
            periods=[period], 
            price_col='mid_price'
        )

print(f"开始计算 {len(factor_names)} 个因子的IC值...")

# 创建IC结果存储结构
ic_results = {}
mean_ic = {}
count_ic = {}

for period in RETURN_PERIODS:
    ic_results[period] = {}
    mean_ic[period] = {}
    count_ic[period] = {}

# 计算每个因子的IC
config = FactorTestConfig(return_periods=RETURN_PERIODS)

for i, factor_name in enumerate(tqdm(factor_names, desc="计算IC")):
    if factor_name not in df.columns:
        print(f"警告: 因子 {factor_name} 不在数据中，跳过")
        continue
    
    try:
        # 计算IC
        ic_df = FactorsTester.calculate_ic(
            df, 
            factor_names=factor_name,
            return_periods=RETURN_PERIODS, 
            method=config.ic_method
        )
        
        if ic_df.empty:
            print(f"警告: 因子 {factor_name} 的IC计算结果为空")
            continue
        
        # 保存每个周期的IC
        for period in RETURN_PERIODS:
            ic_col = f'{factor_name}_{period}period_ic'
            if ic_col in ic_df.columns:
                ic_values = ic_df[ic_col].dropna()
                
                if not ic_values.empty:
                    ic_results[period][factor_name] = ic_values
                    mean_ic[period][factor_name] = ic_values.mean()
                    count_ic[period][factor_name] = len(ic_values)
    except Exception as e:
        print(f"计算因子 {factor_name} 的IC时出错: {str(e)}")

# 创建结果目录
print(f"\n保存IC计算结果...")

# 保存IC均值排序结果
for period in RETURN_PERIODS:
    if mean_ic[period]:
        # 创建均值IC的DataFrame并排序
        mean_ic_df = pd.DataFrame({
            'factor': list(mean_ic[period].keys()),
            'mean_ic': list(mean_ic[period].values()),
            'count': [count_ic[period].get(factor, 0) for factor in mean_ic[period].keys()]
        })
        
        # 按IC绝对值大小排序
        mean_ic_df['abs_mean_ic'] = mean_ic_df['mean_ic'].abs()
        mean_ic_df = mean_ic_df.sort_values('abs_mean_ic', ascending=False)
        
        # 保存结果
        mean_ic_file = os.path.join(output_dir, f'mean_ic_{period}period.csv')
        mean_ic_df.to_csv(mean_ic_file, index=False)
        print(f"保存了{period}期均值IC到: {mean_ic_file}")
        
        # 输出前20个因子
        print(f"\n{period}期IC最大的前20个因子:")
        top_factors = mean_ic_df.head(20)
        for idx, row in top_factors.iterrows():
            print(f"  {row['factor']}: {row['mean_ic']:.4f} (样本数: {row['count']})")

# 保存完整IC序列
for period in RETURN_PERIODS:
    if ic_results[period]:
        # 合并所有因子的IC序列
        all_ic_df = pd.DataFrame()
        
        for factor, ic_series in ic_results[period].items():
            if not ic_series.empty:
                all_ic_df[factor] = ic_series
        
        if not all_ic_df.empty:
            # 保存结果
            ic_file = os.path.join(output_dir, f'ic_series_{period}period.csv')
            all_ic_df.to_csv(ic_file)
            print(f"保存了{period}期IC序列到: {ic_file}")

# 创建所有周期的均值IC汇总
summary_df = pd.DataFrame({'factor': factor_names})

for period in RETURN_PERIODS:
    period_mean_ic = {factor: mean_ic[period].get(factor, np.nan) for factor in factor_names}
    summary_df[f'{period}period_mean_ic'] = summary_df['factor'].map(period_mean_ic)

# 计算所有周期的平均绝对IC
summary_df['avg_abs_ic'] = summary_df[[f'{period}period_mean_ic' for period in RETURN_PERIODS]].abs().mean(axis=1)
summary_df = summary_df.sort_values('avg_abs_ic', ascending=False)

# 保存综合排名结果
summary_file = os.path.join(output_dir, 'factor_ic_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"\n保存了因子IC综合排名到: {summary_file}")

# 输出综合排名前30的因子
print("\n综合IC排名前30的因子:")
top30 = summary_df.head(30)
for idx, row in top30.iterrows():
    ic_values = ", ".join([f"{period}期: {row[f'{period}period_mean_ic']:.4f}" for period in RETURN_PERIODS])
    print(f"  {row['factor']}: {row['avg_abs_ic']:.4f} ({ic_values})")

print("\n处理完成！") 