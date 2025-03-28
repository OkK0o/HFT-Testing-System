import pandas as pd
import numpy as np
import os
import gc
import argparse
import datetime
from typing import List, Dict, Union, Tuple, Optional
from tqdm import tqdm
import traceback
from scipy.stats import spearmanr, pearsonr

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from factor_compute import compute_factors_in_chunks
from fin_data_processor import FinDataProcessor
from factors_test import FactorsTester

def calculate_daily_ic(df: pd.DataFrame, 
                      factor_name: str, 
                      return_periods: List[int] = [1, 5, 10, 20],
                      ic_method: str = 'spearman') -> pd.Series:
    """
    计算单个因子在给定数据上的IC值
    
    Args:
        df: 输入数据
        factor_name: 因子名
        return_periods: 收益率周期列表
        ic_method: IC计算方法，'spearman'或'pearson'
        
    Returns:
        包含不同周期IC值的Series
    """
    ic_values = {}
    
    # 确保因子列存在
    if factor_name not in df.columns:
        return pd.Series(np.nan, index=[f'ic_{p}' for p in return_periods])
    
    # 对每个周期计算IC
    for period in return_periods:
        return_col = f'{period}period_return'
        
        if return_col not in df.columns:
            ic_values[f'ic_{period}'] = np.nan
            continue
        
        # 选择有效数据
        valid_data = df[[factor_name, return_col]].dropna()
        
        if len(valid_data) < 30:  # 最小样本量
            ic_values[f'ic_{period}'] = np.nan
            continue
        
        # 计算IC
        if ic_method == 'spearman':
            ic = spearmanr(valid_data[factor_name], valid_data[return_col])[0]
        else:  # 'pearson'
            ic = pearsonr(valid_data[factor_name], valid_data[return_col])[0]
            
        ic_values[f'ic_{period}'] = ic
    
    return pd.Series(ic_values)

def calculate_factors_returns_and_ic(df: pd.DataFrame,
                                   factor_names: List[str],
                                   return_periods: List[int] = [1, 5, 10, 20],
                                   price_col: str = 'mid_price',
                                   dropna: bool = True,
                                   ic_method: str = 'spearman',
                                   chunk_size: int = None,
                                   overlap_size: int = None,
                                   adjust_sign: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算因子值、收益率和IC，支持分块处理
    
    Args:
        df: 输入数据
        factor_names: 要计算的因子列表
        return_periods: 收益率周期列表
        price_col: 用于计算收益率的价格列
        dropna: 是否删除含有空值的行
        ic_method: IC计算方法，'spearman'或'pearson'
        chunk_size: 分块大小，如果为None则不分块处理
        overlap_size: 分块重叠大小，如果为None则设为最大窗口大小
        adjust_sign: 是否调整因子符号（在分块计算时生效）
        
    Returns:
        Tuple[DataFrame, DataFrame]: (包含因子值和收益率的DataFrame, 包含IC值的DataFrame)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pandas DataFrame")
        
    if df.empty:
        raise ValueError("输入DataFrame为空")
    
    # 注册所有因子
    register_all_factors()
    
    # 确认分类
    tick_factors = []
    minute_factors = []
    
    for factor in factor_names:
        freq = FactorManager.get_factor_frequency(factor)
        if freq == FactorFrequency.TICK:
            tick_factors.append(factor)
        elif freq == FactorFrequency.MINUTE:
            minute_factors.append(factor)
    
    # 保存原始数据的副本
    result_df = df.copy()
    
    # 如果使用分块处理
    if chunk_size is not None:
        print(f"使用分块处理，块大小: {chunk_size}")
        if overlap_size is None:
            # 默认重叠区域为最大窗口大小
            overlap_size = 1200
            print(f"设置默认重叠区域大小: {overlap_size}")
        
        # 对数据进行排序，确保时间顺序正确
        if 'DateTime' in result_df.columns:
            result_df = result_df.sort_values('DateTime')
            
        # 计算所有因子（分块处理）
        print(f"开始分块计算 {len(factor_names)} 个因子...")
        result_df = compute_factors_in_chunks(
            df=result_df,
            factor_names=factor_names,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            adjust_sign=adjust_sign
        )
    else:
        # 常规处理（非分块）
        # 计算Tick频率因子
        if tick_factors:
            print(f"计算 {len(tick_factors)} 个Tick频率因子...")
            tick_df = FactorManager.calculate_factors(df, frequency=FactorFrequency.TICK, factor_names=tick_factors)
            # 合并因子到结果数据
            for factor in tick_factors:
                if factor in tick_df.columns:
                    result_df[factor] = tick_df[factor]
        
        # 计算Minute频率因子
        if minute_factors:
            print(f"计算 {len(minute_factors)} 个Minute频率因子...")
            # 重采样到分钟级别
            data_processor = FinDataProcessor("data")
            minute_df = data_processor.resample_data(df, freq='1min')
            if 'TradDay' not in minute_df.columns and 'DateTime' in minute_df.columns:
                minute_df['TradDay'] = minute_df['DateTime'].dt.date
            
            # 计算因子
            minute_factor_df = FactorManager.calculate_factors(minute_df, frequency=FactorFrequency.MINUTE, factor_names=minute_factors)
            
            # 合并回结果数据
            for factor in minute_factors:
                if factor in minute_factor_df.columns:
                    # 使用DateTime列合并
                    if 'DateTime' in df.columns and 'DateTime' in minute_factor_df.columns:
                        factor_series = minute_factor_df[['DateTime', factor]].dropna()
                        if not factor_series.empty:
                            # 创建一个临时DataFrame进行合并
                            temp_df = result_df[['DateTime']].copy()
                            temp_df = temp_df.merge(factor_series, on='DateTime', how='left')
                            result_df[factor] = temp_df[factor]
    
    # 确保有mid_price列用于计算收益率
    if price_col not in result_df.columns and price_col == 'mid_price':
        if 'AskPrice1' in result_df.columns and 'BidPrice1' in result_df.columns:
            print("计算中间价...")
            result_df['mid_price'] = np.where(
                (result_df['AskPrice1'] == 0) & (result_df['BidPrice1'] == 0),
                np.nan,
                np.where(
                    (result_df['AskPrice1'] == 0) & (result_df['BidPrice1'] != 0),
                    result_df['BidPrice1'],
                    np.where(
                        (result_df['BidPrice1'] == 0) & (result_df['AskPrice1'] != 0),
                        result_df['AskPrice1'],
                        (result_df['AskPrice1'] + result_df['BidPrice1']) / 2
                    )
                )
            )
            result_df['mid_price'] = result_df.groupby('InstruID')['mid_price'].fillna(method='ffill')
    
    # 计算不同周期的收益率
    print("计算收益率...")
    result_df = FactorsTester.calculate_forward_returns(result_df, periods=return_periods, price_col=price_col)
    
    # 去除空值
    if dropna:
        # 确定要检查的列
        check_cols = factor_names.copy()
        for period in return_periods:
            return_col = f'{period}period_return'
            if return_col in result_df.columns:
                check_cols.append(return_col)
        
        # 只保留所有指定列都有值的行
        valid_cols = [col for col in check_cols if col in result_df.columns]
        if valid_cols:
            before_rows = len(result_df)
            result_df = result_df.dropna(subset=valid_cols)
            after_rows = len(result_df)
            print(f"删除空值后，行数从 {before_rows} 减少到 {after_rows} (减少了 {before_rows - after_rows} 行)")
    
    # 计算每个因子的IC值
    ic_df = pd.DataFrame()
    for factor in factor_names:
        if factor in result_df.columns:
            ic_series = calculate_daily_ic(result_df, factor, return_periods, ic_method)
            ic_df[factor] = ic_series
    
    return result_df, ic_df

def load_and_process_daily_data(date_str: str,
                              data_dir: str,
                              factor_names: List[str],
                              return_periods: List[int] = [1, 5, 10, 20],
                              price_col: str = 'mid_price',
                              dropna: bool = True,
                              ic_method: str = 'spearman',
                              chunk_size: int = None,
                              overlap_size: int = None,
                              adjust_sign: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    加载并处理单日数据，支持分块处理
    
    Args:
        date_str: 日期字符串，格式为'YYYYMMDD'
        data_dir: 数据目录
        factor_names: 要计算的因子列表
        return_periods: 收益率周期列表
        price_col: 用于计算收益率的价格列
        dropna: 是否删除含有空值的行
        ic_method: IC计算方法，'spearman'或'pearson'
        chunk_size: 分块大小，如果为None则不分块处理
        overlap_size: 分块重叠大小，如果为None则设为最大窗口大小
        adjust_sign: 是否调整因子符号（在分块计算时生效）
        
    Returns:
        Tuple[Optional[DataFrame], Optional[DataFrame]]: (包含因子值和收益率的DataFrame, 包含IC值的DataFrame)
    """
    try:
        # 构建文件路径
        file_path = os.path.join(data_dir, f"{date_str}.feather")
        
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在")
            return None, None
        
        # 加载数据
        df = pd.read_feather(file_path)
        
        if df.empty:
            print(f"文件 {date_str} 不包含数据")
            return None, None
        
        print(f"处理日期: {date_str}, 数据大小: {df.shape}")
        
        # 计算因子、收益率和IC
        result_df, ic_df = calculate_factors_returns_and_ic(
            df=df,
            factor_names=factor_names,
            return_periods=return_periods,
            price_col=price_col,
            dropna=dropna,
            ic_method=ic_method,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            adjust_sign=adjust_sign
        )
        
        return result_df, ic_df
        
    except Exception as e:
        print(f"处理日期 {date_str} 时出错: {str(e)}")
        print(traceback.format_exc())
        return None, None

def process_date_range(data_dir: str,
                     output_dir: str,
                     start_date: Union[str, datetime.datetime] = None,
                     end_date: Union[str, datetime.datetime] = None,
                     factor_names: List[str] = None,
                     return_periods: List[int] = [1, 5, 10, 20],
                     price_col: str = 'mid_price',
                     dropna: bool = True,
                     ic_method: str = 'spearman',
                     chunk_size: int = None,
                     overlap_size: int = None,
                     adjust_sign: bool = True,
                     overwrite: bool = False) -> Dict[str, pd.DataFrame]:
    """
    处理日期范围内的数据，计算因子值、收益率和IC，支持分块处理
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        start_date: 开始日期，如果为None则使用可用的最早日期
        end_date: 结束日期，如果为None则使用可用的最晚日期
        factor_names: 要计算的因子列表，如果为None则计算所有因子
        return_periods: 要计算的收益率周期列表
        price_col: 用于计算收益率的价格列
        dropna: 是否删除含有空值的行
        ic_method: IC计算方法，'spearman'或'pearson'
        chunk_size: 分块大小，如果为None则不分块处理
        overlap_size: 分块重叠大小，如果为None则设为最大窗口大小
        adjust_sign: 是否调整因子符号（在分块计算时生效）
        overwrite: 是否覆盖已存在的文件
        
    Returns:
        Dict[str, pd.DataFrame]: 字典，包含IC汇总信息
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
    
    # 创建已处理日期的列表和IC结果存储
    processed_dates = []
    all_ic_results = []
    
    # 处理每个日期文件
    for date, file in tqdm(date_files, desc="处理日期"):
        date_str = date.strftime('%Y%m%d')
        output_file = os.path.join(output_dir, f"factors_returns_{date_str}.feather")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file) and not overwrite:
            print(f"文件 {output_file} 已存在，跳过")
            processed_dates.append(date_str)
            continue
        
        # 加载并处理数据
        result_df, ic_df = load_and_process_daily_data(
            date_str=os.path.splitext(file)[0],
            data_dir=data_dir,
            factor_names=factor_names,
            return_periods=return_periods,
            price_col=price_col,
            dropna=dropna,
            ic_method=ic_method,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            adjust_sign=adjust_sign
        )
        
        if result_df is not None and not result_df.empty:
            # 保存结果
            print(f"保存结果到 {output_file}")
            result_df.to_feather(output_file)
            processed_dates.append(date_str)
            
            # 保存IC结果
            if ic_df is not None and not ic_df.empty:
                ic_df['date'] = date
                all_ic_results.append(ic_df)
        
        # 释放内存
        if 'result_df' in locals():
            del result_df
        if 'ic_df' in locals():
            del ic_df
        gc.collect()
    
    # 保存处理成功的日期列表
    dates_file = os.path.join(output_dir, "processed_dates.txt")
    with open(dates_file, 'w') as f:
        for date in processed_dates:
            f.write(f"{date}\n")
    
    # 处理IC结果
    ic_summary = {}
    if all_ic_results:
        # 合并所有日期的IC结果
        combined_ic = pd.concat(all_ic_results, ignore_index=True)
        combined_ic.to_csv(os.path.join(output_dir, "daily_ic.csv"), index=False)
        
        # 计算平均IC
        ic_mean = combined_ic.drop(columns=['date']).mean()
        ic_std = combined_ic.drop(columns=['date']).std()
        ic_ir = ic_mean / ic_std
        
        # 创建IC汇总表
        ic_summary = {
            'ic_mean': ic_mean.to_dict(),
            'ic_std': ic_std.to_dict(),
            'ic_ir': ic_ir.to_dict(),
            'daily_ic': combined_ic.to_dict()
        }
        
        # 保存IC汇总
        with open(os.path.join(output_dir, "ic_summary.csv"), 'w') as f:
            f.write("factor,")
            for period in return_periods:
                f.write(f"ic_{period}_mean,ic_{period}_std,ic_{period}_ir,")
            f.write("\n")
            
            for factor in factor_names:
                if factor in ic_mean:
                    f.write(f"{factor},")
                    for period in return_periods:
                        col = f"ic_{period}"
                        mean_val = ic_mean.get(factor, {}).get(col, np.nan)
                        std_val = ic_std.get(factor, {}).get(col, np.nan)
                        ir_val = ic_ir.get(factor, {}).get(col, np.nan)
                        f.write(f"{mean_val:.4f},{std_val:.4f},{ir_val:.4f},")
                    f.write("\n")
    
    print(f"\n处理完成! 成功处理了 {len(processed_dates)}/{len(date_files)} 个日期")
    print(f"结果保存在: {output_dir}")
    print(f"处理成功的日期列表保存在: {dates_file}")
    
    return ic_summary

def compute_and_save_factors_with_returns(
    data_dir: str,
    output_dir: str,
    start_date: Union[str, datetime.datetime] = None,
    end_date: Union[str, datetime.datetime] = None,
    factor_names: List[str] = None,
    return_periods: List[int] = [1, 5, 10, 20],
    price_col: str = 'mid_price',
    dropna: bool = True,
    overwrite: bool = False,
    ic_method: str = 'spearman',
    chunk_size: int = None,
    overlap_size: int = None,
    adjust_sign: bool = True
):
    """
    按天计算因子值、不同周期的收益率、IC值，去除空行，并保存，支持分块处理
    
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
        ic_method: IC计算方法，'spearman'或'pearson'
        chunk_size: 分块大小，如果为None则不分块处理
        overlap_size: 分块重叠大小，如果为None则设为最大窗口大小
        adjust_sign: 是否调整因子符号（在分块计算时生效）
    """
    return process_date_range(
        data_dir=data_dir,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        factor_names=factor_names,
        return_periods=return_periods,
        price_col=price_col,
        dropna=dropna,
        ic_method=ic_method,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        adjust_sign=adjust_sign,
        overwrite=overwrite
    )

def load_and_combine_processed_data(
    output_dir: str,
    start_date: Union[str, datetime.datetime] = None,
    end_date: Union[str, datetime.datetime] = None,
    limit_rows: int = None
) -> pd.DataFrame:
    """
    加载并合并处理好的数据
    
    Args:
        output_dir: 处理后的数据目录
        start_date: 开始日期，如果为None则使用可用的最早日期
        end_date: 结束日期，如果为None则使用可用的最晚日期
        limit_rows: 每个文件读取的最大行数
        
    Returns:
        pd.DataFrame: 合并后的数据
    """
    # 获取所有处理过的日期
    dates_file = os.path.join(output_dir, "processed_dates.txt")
    if not os.path.exists(dates_file):
        # 如果没有日期列表文件，则尝试从目录中读取
        files = [f for f in os.listdir(output_dir) if f.startswith("factors_returns_") and f.endswith(".feather")]
        processed_dates = [f[16:24] for f in files]  # 提取日期部分
    else:
        with open(dates_file, 'r') as f:
            processed_dates = [line.strip() for line in f]
    
    # 日期范围过滤
    if start_date is not None:
        start_date = pd.to_datetime(start_date).strftime('%Y%m%d')
        processed_dates = [d for d in processed_dates if d >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
        processed_dates = [d for d in processed_dates if d <= end_date]
    
    # 排序日期
    processed_dates.sort()
    
    print(f"加载 {len(processed_dates)} 个日期的数据...")
    
    # 逐个加载并合并
    all_data = []
    for date_str in tqdm(processed_dates, desc="加载数据"):
        file_path = os.path.join(output_dir, f"factors_returns_{date_str}.feather")
        if os.path.exists(file_path):
            try:
                if limit_rows:
                    # 只读取部分行以节省内存
                    df = pd.read_feather(file_path, nthreads=1)
                    if len(df) > limit_rows:
                        df = df.sample(n=limit_rows, random_state=42)
                else:
                    df = pd.read_feather(file_path, nthreads=1)
                
                # 确保有日期列
                if 'date' not in df.columns and 'TradDay' in df.columns:
                    df['date'] = df['TradDay']
                elif 'date' not in df.columns and 'DateTime' in df.columns:
                    df['date'] = df['DateTime'].dt.date
                
                all_data.append(df)
                
                # 释放内存
                del df
                gc.collect()
                
            except Exception as e:
                print(f"加载 {file_path} 时出错: {str(e)}")
    
    if not all_data:
        raise ValueError("没有成功加载任何数据")
    
    # 合并数据
    print("合并所有数据...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 释放内存
    del all_data
    gc.collect()
    
    return combined_df

def get_ic_summary(output_dir: str) -> pd.DataFrame:
    """
    获取IC汇总信息
    
    Args:
        output_dir: 处理后的数据目录
        
    Returns:
        pd.DataFrame: IC汇总数据
    """
    ic_summary_path = os.path.join(output_dir, "ic_summary.csv")
    if os.path.exists(ic_summary_path):
        return pd.read_csv(ic_summary_path)
    
    daily_ic_path = os.path.join(output_dir, "daily_ic.csv")
    if os.path.exists(daily_ic_path):
        daily_ic = pd.read_csv(daily_ic_path)
        
        # 汇总IC信息
        # 忽略'date'列
        factors = [col for col in daily_ic.columns if col != 'date']
        
        # 计算每个因子和周期的统计信息
        results = []
        for factor in factors:
            factor_data = daily_ic[[factor, 'date']].dropna()
            
            if not factor_data.empty:
                mean_val = factor_data[factor].mean()
                std_val = factor_data[factor].std()
                ir_val = mean_val / std_val if std_val != 0 else np.nan
                
                # 提取周期信息
                parts = factor.split('_')
                if len(parts) >= 2 and parts[0] == 'ic':
                    period = parts[1]
                    results.append({
                        'factor': factor,
                        'period': period,
                        'mean': mean_val,
                        'std': std_val,
                        'ir': ir_val
                    })
        
        return pd.DataFrame(results)
    
    raise FileNotFoundError(f"在 {output_dir} 中没有找到IC汇总信息")

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
    parser.add_argument('--ic-method', type=str, default='spearman', choices=['spearman', 'pearson'], help='IC计算方法')
    parser.add_argument('--chunk-size', type=int, default=None, help='分块大小，如果不指定则不分块处理')
    parser.add_argument('--overlap-size', type=int, default=None, help='分块重叠大小，如果不指定则设为最大窗口大小')
    parser.add_argument('--adjust-sign', action='store_true', default=True, help='是否调整因子符号（在分块计算时生效）')
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
        overwrite=args.overwrite,
        ic_method=args.ic_method,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        adjust_sign=args.adjust_sign
    )

if __name__ == "__main__":
    main() 