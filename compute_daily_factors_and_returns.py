import pandas as pd
import numpy as np
import os
import gc
import argparse
import datetime
from typing import List, Dict, Union
from tqdm import tqdm
import traceback

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from fin_data_processor import FinDataProcessor
from factors_test import FactorsTester

def compute_and_save_factors_with_returns(
    data_dir: str,
    output_dir: str,
    start_date: Union[str, datetime.datetime] = None,
    end_date: Union[str, datetime.datetime] = None,
    factor_names: List[str] = None,
    return_periods: List[int] = [1, 5, 10, 20],
    price_col: str = 'mid_price',
    dropna: bool = True,
    overwrite: bool = False
):
    """
    按天计算因子值和不同周期的收益率，去除空行，并保存
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        start_date: 开始日期，如果为None则使用可用的最早日期
        end_date: 结束日期，如果为None则使用可用的最晚日期
        factor_names: 要计算的因子列表，如果为None则计算所有因子
        return_periods: 要计算的收益率周期列表
        price_col: 用于计算收益率的价格列
        dropna: 是否删除含有空值的行
        overwrite: 是否覆盖已存在的文件
    """
    # 注册所有因子
    register_all_factors()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果factor_names为None，获取所有可用因子
    if factor_names is None:
        factor_info = FactorManager.get_factor_info()
        factor_names = factor_info['name'].tolist()
        print(f"将计算所有 {len(factor_names)} 个因子")
    else:
        print(f"将计算指定的 {len(factor_names)} 个因子")
    
    # 获取所有数据文件
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.feather')])
    
    if not files:
        raise ValueError(f"在目录 {data_dir} 中没有找到数据文件")
    
    # 处理日期范围
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
    
    # 处理日期相关逻辑
    date_files = []
    for file in files:
        try:
            # 尝试从文件名提取日期
            date_str = os.path.splitext(file)[0]  # 去除文件扩展名
            if date_str.isdigit() and len(date_str) == 8:  # 格式如20220101
                date = pd.to_datetime(date_str)
                
                # 检查日期范围
                if start_date is not None and date < start_date:
                    continue
                if end_date is not None and date > end_date:
                    continue
                
                date_files.append((date, file))
        except:
            # 如果无法解析日期，跳过
            continue
    
    if not date_files:
        raise ValueError("没有在指定日期范围内找到数据文件")
    
    # 按日期排序
    date_files.sort(key=lambda x: x[0])
    
    print(f"找到 {len(date_files)} 个符合条件的日期文件")
    print(f"将计算 {', '.join([str(p) for p in return_periods])} 周期的收益率")
    
    # 创建已处理日期的列表
    processed_dates = []
    
    # 处理每个日期文件
    for date, file in tqdm(date_files, desc="处理日期"):
        date_str = date.strftime('%Y%m%d')
        output_file = os.path.join(output_dir, f"factors_returns_{date_str}.feather")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file) and not overwrite:
            print(f"文件 {output_file} 已存在，跳过")
            processed_dates.append(date_str)
            continue
        
        try:
            # 加载数据
            file_path = os.path.join(data_dir, file)
            df = pd.read_feather(file_path)
            
            if df.empty:
                print(f"文件 {file} 不包含数据，跳过")
                continue
            
            print(f"\n处理日期: {date_str}, 数据大小: {df.shape}")
            
            # 计算因子值
            tick_factors = []
            minute_factors = []
            
            # 将因子按频率分类
            for factor in factor_names:
                freq = FactorManager.get_factor_frequency(factor)
                if freq == FactorFrequency.TICK:
                    tick_factors.append(factor)
                elif freq == FactorFrequency.MINUTE:
                    minute_factors.append(factor)
            
            # 计算Tick频率因子
            if tick_factors:
                print(f"计算 {len(tick_factors)} 个Tick频率因子")
                tick_df = FactorManager.calculate_factors(df, frequency=FactorFrequency.TICK, factor_names=tick_factors)
                # 合并因子到原始数据
                for factor in tick_factors:
                    if factor in tick_df.columns:
                        df[factor] = tick_df[factor]
            
            # 计算Minute频率因子
            if minute_factors:
                print(f"计算 {len(minute_factors)} 个Minute频率因子")
                # 重采样到分钟级别
                data_processor = FinDataProcessor("data")
                minute_df = data_processor.resample_data(df, freq='1min')
                if 'TradDay' not in minute_df.columns and 'DateTime' in minute_df.columns:
                    minute_df['TradDay'] = minute_df['DateTime'].dt.date
                
                # 计算因子
                minute_factor_df = FactorManager.calculate_factors(minute_df, frequency=FactorFrequency.MINUTE, factor_names=minute_factors)
                
                # 合并回原始数据
                for factor in minute_factors:
                    if factor in minute_factor_df.columns:
                        # 使用DateTime列合并
                        if 'DateTime' in df.columns and 'DateTime' in minute_factor_df.columns:
                            factor_series = minute_factor_df[['DateTime', factor]].dropna()
                            if not factor_series.empty:
                                # 创建一个临时DataFrame进行合并
                                temp_df = df[['DateTime']].copy()
                                temp_df = temp_df.merge(factor_series, on='DateTime', how='left')
                                df[factor] = temp_df[factor]
            
            # 确保有mid_price列用于计算收益率
            if price_col not in df.columns and price_col == 'mid_price':
                if 'AskPrice1' in df.columns and 'BidPrice1' in df.columns:
                    print("计算中间价...")
                    df['mid_price'] = np.where(
                        (df['AskPrice1'] == 0) & (df['BidPrice1'] == 0),
                        np.nan,
                        np.where(
                            (df['AskPrice1'] == 0) & (df['BidPrice1'] != 0),
                            df['BidPrice1'],
                            np.where(
                                (df['BidPrice1'] == 0) & (df['AskPrice1'] != 0),
                                df['AskPrice1'],
                                (df['AskPrice1'] + df['BidPrice1']) / 2
                            )
                        )
                    )
                    df['mid_price'] = df.groupby('InstruID')['mid_price'].fillna(method='ffill')
            
            # 计算不同周期的收益率
            print("计算收益率...")
            df = FactorsTester.calculate_forward_returns(df, periods=return_periods, price_col=price_col)
            
            # 去除空值
            if dropna:
                # 确定要检查的列
                check_cols = factor_names.copy()
                for period in return_periods:
                    check_cols.append(f'{period}period_return')
                
                # 只保留所有因子和收益率都有值的行
                valid_cols = [col for col in check_cols if col in df.columns]
                before_rows = len(df)
                df = df.dropna(subset=valid_cols)
                after_rows = len(df)
                print(f"删除空值后，行数从 {before_rows} 减少到 {after_rows} (减少了 {before_rows - after_rows} 行)")
            
            # 保存结果
            print(f"保存结果到 {output_file}")
            df.to_feather(output_file)
            processed_dates.append(date_str)
            
            # 释放内存
            del df
            if 'tick_df' in locals():
                del tick_df
            if 'minute_df' in locals():
                del minute_df
            if 'minute_factor_df' in locals():
                del minute_factor_df
            gc.collect()
            
        except Exception as e:
            print(f"处理日期 {date_str} 时出错: {str(e)}")
            print(traceback.format_exc())
    
    # 保存处理成功的日期列表
    dates_file = os.path.join(output_dir, "processed_dates.txt")
    with open(dates_file, 'w') as f:
        for date in processed_dates:
            f.write(f"{date}\n")
    
    print(f"\n处理完成! 成功处理了 {len(processed_dates)}/{len(date_files)} 个日期")
    print(f"结果保存在: {output_dir}")
    print(f"处理成功的日期列表保存在: {dates_file}")

def main():
    parser = argparse.ArgumentParser(description='按天计算因子值和收益率并保存')
    parser.add_argument('--data-dir', type=str, default='data', help='原始数据目录')
    parser.add_argument('--output-dir', type=str, default='factors_returns_by_date', help='输出目录')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--factors', type=str, default=None, help='要计算的因子，以逗号分隔，如果不指定则计算所有因子')
    parser.add_argument('--returns', type=str, default='1,5,10,20', help='要计算的收益率周期，以逗号分隔')
    parser.add_argument('--price-col', type=str, default='mid_price', help='用于计算收益率的价格列')
    parser.add_argument('--keep-na', action='store_true', help='是否保留含有空值的行')
    parser.add_argument('--overwrite', action='store_true', help='是否覆盖已存在的文件')
    args = parser.parse_args()
    
    # 解析因子和收益率列表
    factor_names = None
    if args.factors:
        factor_names = [f.strip() for f in args.factors.split(',')]
    
    return_periods = [int(p.strip()) for p in args.returns.split(',')]
    
    # 调用主函数
    compute_and_save_factors_with_returns(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        factor_names=factor_names,
        return_periods=return_periods,
        price_col=args.price_col,
        dropna=not args.keep_na,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main() 