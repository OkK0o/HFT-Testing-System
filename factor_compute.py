# factor_example.py

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from fin_data_processor import FinDataProcessor
from factors_test import FactorsTester, FactorTestConfig
import pandas as pd
import datetime
from typing import List, Dict, Union, Iterator
import os
import numpy as np
from scipy import stats

def run_factor_compute(df: pd.DataFrame, 
                      factor_name: str = None, 
                      config: FactorTestConfig = None,
                      save_path: str = "factor_test_results",
                      return_periods: List[int] = None,
                      generate_images: bool = False,
                      adjust_sign: bool = True) -> Dict:
    """
    运行因子测试
    
    Args:
        df: 输入的DataFrame数据
        factor_name: 要测试的因子名称，如果为None则显示所有因子
        config: 测试配置，如果为None则使用默认配置
        save_path: 结果保存路径
        return_periods: 自定义收益率计算周期列表，如[1, 5, 10, 20]
        generate_images: 是否生成图片
        adjust_sign: 是否根据IC调整因子符号
        
    Returns:
        包含测试结果的字典
    """
    register_all_factors()
    
    if factor_name is not None:
        factor_freq = FactorManager.get_factor_frequency(factor_name)
        if factor_freq is None:
            print(f"\n错误：因子 '{factor_name}' 不存在")
            print("\n可用的因子列表：")
            print(FactorManager.get_factor_info())
            return None
            
        # 使用自定义收益率周期或默认值
        periods = return_periods if return_periods is not None else [1, 5, 10, 20]
        print(f"\n计算收益率周期: {periods}")
        
        if config is None:
            config = FactorTestConfig(return_periods=periods)
        else:
            config.return_periods = periods
            
        tester = FactorsTester(
            save_path=save_path,
            config=config
        )
            
        data = df.copy()
        print(f"\n原始数据形状: {data.shape}")
        
        try:
            if factor_freq == FactorFrequency.TICK:
                factor_df = FactorManager.calculate_factors(df, frequency=FactorFrequency.TICK, factor_names=factor_name)
                if factor_name in factor_df.columns:
                    data[factor_name] = factor_df[factor_name]
            else:
                data_processor = FinDataProcessor("data")
                minute_df = data_processor.resample_data(df, freq='1min')
                minute_df['TradDay'] = minute_df['DateTime'].dt.date
                factor_df = FactorManager.calculate_factors(minute_df, frequency=FactorFrequency.MINUTE, factor_names=factor_name)
                if factor_name in factor_df.columns:
                    data = data.merge(factor_df[['DateTime', factor_name]], 
                                    on='DateTime', 
                                    how='left')
            
            print(f"\n添加因子后数据形状: {data.shape}")
            data = FactorsTester.calculate_forward_returns(data, periods=periods)
            print(f"\n添加收益率后数据形状: {data.shape}")
            
            # 运行因子测试，传入generate_images参数
            results = tester.test_single_factor(factor_name, data, config, generate_images=generate_images)
            
            # 如果需要调整符号，根据IC值调整因子符号
            if adjust_sign and 'ic_series' in results:
                ic_series = results['ic_series']
                # 计算每个周期的平均IC
                mean_ics = ic_series.mean()
                # 如果IC为负，调整因子符号
                for period in periods:
                    if mean_ics[f'{period}period_ic'] < 0:
                        data[factor_name] = -data[factor_name]
                        print(f"\n调整因子 {factor_name} 的符号（周期 {period}）")
                
                # 重新运行测试
                results = tester.test_single_factor(factor_name, data, config, generate_images=generate_images)
            
            # 汇总结果到DataFrame
            summary_data = []
            for period in periods:
                period_metrics = results['evaluation']['metrics'].copy()  # 使用copy避免修改原始数据
                period_metrics.update({
                    'factor': factor_name,
                    'period': period,
                    'is_effective': results['evaluation']['is_effective'],
                    'score': results['evaluation']['score']
                })
                summary_data.append(period_metrics)
            
            # 生成图片
            if generate_images:
                # IC时间序列图
                if 'ic_series' in results:
                    FactorsTester.plot_ic_series(
                        {factor_name: results['ic_series']},
                        factor_names=factor_name,
                        periods=periods
                    )
                
                # 分位数回测结果图
                if 'quantile_results' in results:
                    for period in periods:
                        if period in results['quantile_results'].get('quantile_returns', {}):
                            FactorsTester.plot_quantile_results(
                                results=results['quantile_results'],
                                period=period,
                                factor_name=factor_name
                            )
            
            results['summary'] = pd.DataFrame(summary_data)
            return results
            
        except Exception as e:
            print(f"\n因子测试过程中出错:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            return None
            
    else:
        print("\n=== 所有已注册因子 ===")
        all_factors = FactorManager.get_factor_info()
        print(all_factors)
        return None

def compute_factor(df: pd.DataFrame, 
                   factor_name: str) -> pd.DataFrame:
    """
    仅计算因子，不进行测试，并将因子值添加回原始数据
    
    Args:
        df: 输入的DataFrame数据
        factor_name: 要计算的因子名称
                
    Returns:
        添加了因子值的原始DataFrame
    """
    register_all_factors()
    factor_freq = FactorManager.get_factor_frequency(factor_name)
    if factor_freq is None:
        print(f"\n错误：因子 '{factor_name}' 不存在")
        print("\n可用的因子列表：")
        print(FactorManager.get_factor_info())
        return df
    original_index = df.index.copy()
    result_df = df.copy()
    try:
        if factor_freq == FactorFrequency.TICK:
            factor_df = FactorManager.calculate_factors(df, frequency=FactorFrequency.TICK, factor_names=factor_name)
            if factor_name in factor_df.columns:
                result_df[factor_name] = factor_df[factor_name]
        else:
            data_processor = FinDataProcessor("data")
            minute_df = data_processor.resample_data(df, freq='1min')
            minute_df['TradDay'] = minute_df['DateTime'].dt.date
            factor_df = FactorManager.calculate_factors(minute_df, frequency=FactorFrequency.MINUTE, factor_names=factor_name)
            if factor_name in factor_df.columns:
                result_df = result_df.merge(factor_df[['DateTime', factor_name]], 
                                          on='DateTime', 
                                          how='left')
        result_df = result_df.loc[original_index]
        print(f"因子 {factor_name} 计算完成")
    except Exception as e:
        print(f"计算因子 {factor_name} 时出错: {str(e)}")
        return df
    
    return result_df

def compute_multiple_factors(df: pd.DataFrame, 
                           factor_names: List[str],
                           adjust_sign: bool = True) -> pd.DataFrame:
    """
    批量计算多个因子
    
    Args:
        df: 输入的DataFrame数据
        factor_names: 要计算的因子名称列表
        adjust_sign: 是否根据IC调整因子符号
        
    Returns:
        添加了所有因子值的原始DataFrame
    """
    result_df = df.copy()
    print(f"\n开始计算 {len(factor_names)} 个因子...")
    
    # 先计算所有因子
    for factor_name in factor_names:
        result_df = compute_factor(result_df, factor_name)
    
    print("\n所有因子计算完成!")
    return result_df

def compute_factors_in_chunks(df: pd.DataFrame, 
                             factor_names: List[str], 
                             chunk_size: int = 500000, 
                             overlap_size: int = 100000,
                             adjust_sign: bool = False,
                             return_periods: List[int] = None) -> pd.DataFrame:
    """
    分块计算因子，用于处理大规模数据，并解决滑动窗口因子边界问题
    
    Args:
        df: 输入的DataFrame数据
        factor_names: 要计算的因子名称列表
        chunk_size: 每块数据的大小
        overlap_size: 块之间的重叠区域大小，用于解决历史滑动窗口因子边界问题
        adjust_sign: 是否根据IC调整因子符号
        return_periods: 用于计算IC的收益率周期，默认为[10]
        
    Returns:
        添加了所有因子的DataFrame
    """
    # 注册所有因子
    register_all_factors()
    
    # 设置默认的收益率周期
    if return_periods is None:
        return_periods = [10]
    
    # 确保数据是按时间排序的
    if 'DateTime' in df.columns:
        df = df.sort_values(['DateTime', 'InstruID']).reset_index(drop=True)
    elif 'TradDay' in df.columns and 'UpdateTime' in df.columns:
        df = df.sort_values(['TradDay', 'UpdateTime', 'InstruID']).reset_index(drop=True)
    elif 'date' in df.columns:
        df = df.sort_values(['date', 'InstruID']).reset_index(drop=True)
    
    # 确保重叠区域大小合理
    if overlap_size >= chunk_size:
        overlap_size = chunk_size // 2
        print(f"重叠区域大小调整为: {overlap_size}")
    
    # 获取数据集大小
    total_rows = len(df)
    print(f"数据集总行数: {total_rows}")
    print(f"分块大小: {chunk_size}, 重叠区域: {overlap_size}")
    
    # 计算需要处理的块数
    num_chunks = (total_rows - overlap_size) // (chunk_size - overlap_size)
    if (total_rows - overlap_size) % (chunk_size - overlap_size) > 0:
        num_chunks += 1
    
    print(f"需要处理的块数: {num_chunks}")
    
    # 初始化结果数据框
    final_df = df.copy()
    # 用于存储每个因子的值
    factor_values = {factor: pd.Series(index=df.index, dtype='float64') for factor in factor_names}
    
    # 获取因子信息
    factor_info = FactorManager.get_factor_info()
    
    # 按块处理数据
    for chunk_idx in range(num_chunks):
        # 计算当前块的起始索引和结束索引
        start_idx = chunk_idx * (chunk_size - overlap_size)
        end_idx = min(start_idx + chunk_size, total_rows)
        
        # 对第一个块，始终从0开始
        if chunk_idx == 0:
            start_idx = 0
        
        # 提取当前块的数据
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        # 输出处理进度
        print(f"\n处理第 {chunk_idx + 1}/{num_chunks} 块数据 (索引 {start_idx} 到 {end_idx-1}), 大小: {len(chunk_df)} 行")
        
        # 计算当前块的因子
        chunk_with_factors = compute_multiple_factors(chunk_df, factor_names, adjust_sign=False)
        
        # 如果需要调整符号，在当前块上直接计算和调整
        if adjust_sign:
            try:
                print(f"调整第 {chunk_idx + 1} 块的因子符号...")
                # 计算前向收益率
                test_df = FactorsTester.calculate_forward_returns(chunk_with_factors, periods=return_periods)
                
                # 为每个因子计算IC值并调整符号
                config = FactorTestConfig(return_periods=return_periods)
                for factor_name in factor_names:
                    if factor_name in chunk_with_factors.columns:
                        # 计算IC
                        ic_df = FactorsTester.calculate_ic(
                            test_df, 
                            factor_names=factor_name,
                            return_periods=return_periods, 
                            method=config.ic_method
                        )
                        
                        # 获取平均IC
                        if not ic_df.empty:
                            ic_col = f'{factor_name}_{return_periods[0]}period_ic'
                            if ic_col in ic_df.columns:
                                mean_ic = ic_df[ic_col].mean()
                                
                                # 如果IC为负，调整因子符号
                                if mean_ic < 0:
                                    chunk_with_factors[factor_name] = -chunk_with_factors[factor_name]
                                    print(f"  调整因子 {factor_name} 的符号 (块内平均IC: {mean_ic:.4f})")
            except Exception as e:
                print(f"调整第 {chunk_idx + 1} 块因子符号时出错: {str(e)}")
                print("继续使用未调整的因子值")
        
        # 有效数据区域计算：排除头部重叠区域
        valid_start_idx = 0
        valid_end_idx = len(chunk_df)
        
        # 确定有效区域 (排除重叠区域的开始部分，第一个块除外)
        # 因为大多数因子只依赖历史数据，头部可能缺乏足够的历史计算窗口
        if chunk_idx > 0:
            valid_start_idx = overlap_size
        
        # 不排除尾部区域，因为大多数因子不依赖未来数据
        
        # 提取有效区域的因子值并更新最终结果
        for factor in factor_names:
            if factor in chunk_with_factors.columns:
                # 取当前块的有效区域
                valid_chunk_indices = chunk_df.index[valid_start_idx:valid_end_idx]
                original_indices = df.index[start_idx + valid_start_idx:start_idx + valid_end_idx]
                
                # 将有效区域的因子值赋给最终结果
                if len(valid_chunk_indices) > 0:
                    # 确保索引匹配
                    factor_values[factor].loc[original_indices] = chunk_with_factors.loc[valid_chunk_indices, factor].values
        
        # 释放内存
        del chunk_df, chunk_with_factors
        import gc
        gc.collect()
        print(f"第 {chunk_idx + 1} 块处理完成，已释放内存")
    
    # 将计算的因子添加到最终结果中
    for factor in factor_names:
        if not factor_values[factor].isna().all():  # 确保有计算结果
            final_df[factor] = factor_values[factor]
            print(f"因子 {factor} 添加完成，非空值: {factor_values[factor].count()}/{len(factor_values[factor])}")
    
    print("\n分块因子计算完成!")
    return final_df

def adjust_factor_signs_in_chunks(df: pd.DataFrame, 
                                factor_names: List[str] = None, 
                                return_periods: List[int] = None,
                                chunk_size: int = 500000) -> pd.DataFrame:
    """
    分块调整因子符号，避免内存溢出
    
    Args:
        df: 包含因子的DataFrame数据
        factor_names: 要调整的因子列表，如果为None则调整所有可用因子
        return_periods: 用于计算IC的收益率周期，默认为[10]
        chunk_size: 每次处理的数据块大小
        
    Returns:
        调整符号后的DataFrame
    """
    result_df = df.copy()
    
    # 如果未指定因子，则获取所有已计算的因子
    if factor_names is None:
        factor_info = FactorManager.get_factor_info()
        all_factors = factor_info['name'].tolist()
        factor_names = [col for col in df.columns if col in all_factors]
    
    # 使用默认收益率周期
    if return_periods is None:
        return_periods = [10]
        
    print(f"\n开始分块调整 {len(factor_names)} 个因子的符号...")
    
    # 为每个因子确定符号调整方向
    factor_signs = {}  # 存储每个因子是否需要反转符号
    
    # 获取数据集大小
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    # 按因子分别处理，降低内存压力
    for factor_name in factor_names:
        if factor_name not in result_df.columns:
            print(f"警告: 因子 {factor_name} 不在数据中，跳过")
            continue
            
        # 计算抽样的IC值
        ic_values = []
        
        # 对每个块计算IC，然后合并结果
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            
            chunk_df = result_df.iloc[start_idx:end_idx].copy()
            
            # 首先计算前向收益率 (仅处理当前块)
            try:
                chunk_df = FactorsTester.calculate_forward_returns(chunk_df, periods=return_periods)
                
                # 计算IC (仅使用当前块)
                config = FactorTestConfig(return_periods=return_periods)
                ic_df = FactorsTester.calculate_ic(
                    chunk_df, 
                    factor_names=factor_name,
                    return_periods=return_periods, 
                    method=config.ic_method
                )
                
                if not ic_df.empty:
                    ic_col = f'{factor_name}_{return_periods[0]}period_ic'
                    if ic_col in ic_df.columns:
                        # 收集IC值
                        ic_values.extend(ic_df[ic_col].dropna().tolist())
                
            except Exception as e:
                print(f"计算因子 {factor_name} 的IC值时出错: {str(e)}")
                continue
                
            # 释放内存
            del chunk_df
            gc.collect()
        
        # 根据平均IC值确定是否需要调整符号
        if ic_values:
            mean_ic = sum(ic_values) / len(ic_values)
            if mean_ic < 0:
                factor_signs[factor_name] = -1
                print(f"因子 {factor_name} 需要反转符号 (平均IC: {mean_ic:.4f})")
            else:
                factor_signs[factor_name] = 1
                print(f"因子 {factor_name} 保持原符号 (平均IC: {mean_ic:.4f})")
    
    # 应用符号调整
    for factor_name, sign in factor_signs.items():
        if sign == -1:
            result_df[factor_name] = -result_df[factor_name]
            print(f"已反转因子 {factor_name} 的符号")
    
    print(f"\n分块调整因子符号完成!")        
    return result_df

def compute_factors_by_contract(df: pd.DataFrame, 
                              factor_names: List[str], 
                              adjust_sign: bool = False) -> pd.DataFrame:
    """
    按合约分组计算因子，适用于大规模数据处理
    
    Args:
        df: 输入的DataFrame数据
        factor_names: 要计算的因子名称列表
        adjust_sign: 是否根据IC调整因子符号
        
    Returns:
        添加了所有因子的DataFrame
    """
    # 注册所有因子
    register_all_factors()
    
    # 获取所有唯一合约
    contracts = df['InstruID'].unique()
    print(f"共有 {len(contracts)} 个合约需要处理")
    
    # 初始化结果数据框
    result_df = df.copy()
    
    # 按合约处理数据
    for i, contract in enumerate(contracts):
        print(f"\n处理合约 {i+1}/{len(contracts)}: {contract}")
        
        # 提取当前合约的数据
        contract_df = df[df['InstruID'] == contract].copy()
        
        try:
            # 计算当前合约的因子
            contract_with_factors = compute_multiple_factors(contract_df, factor_names, adjust_sign=False)
            
            # 将计算结果更新回结果数据框
            for factor in factor_names:
                if factor in contract_with_factors.columns:
                    result_df.loc[contract_df.index, factor] = contract_with_factors[factor].values
            
        except Exception as e:
            print(f"计算合约 {contract} 的因子时出错: {str(e)}")
        
        # 释放内存
        del contract_df
        import gc
        gc.collect()
    
    # 如果需要调整符号
    if adjust_sign:
        print("\n调整因子符号...")
        result_df = adjust_factor_signs(result_df, factor_names=factor_names)
    
    print("\n按合约计算因子完成!")
    return result_df

def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有已注册的因子
    
    Args:
        df: 输入的DataFrame数据
        
    Returns:
        添加了所有因子值的原始DataFrame
    """
    register_all_factors()
    factor_info = FactorManager.get_factor_info()
    all_factors = factor_info['name'].tolist()
    print(f"\n开始计算所有因子 (共 {len(all_factors)} 个)...")
    print("\n因子列表：")
    for category in factor_info['category'].unique():
        category_factors = factor_info[factor_info['category'] == category]['name'].tolist()
        print(f"\n{category}类因子 ({len(category_factors)}个):")
        print(", ".join(category_factors))
    result_df = compute_multiple_factors(df, all_factors)
    factor_columns = [col for col in result_df.columns if col in all_factors]
    print(f"\n计算完成! 共添加 {len(factor_columns)} 个因子列")
    
    return result_df

def adjust_factor_signs(df: pd.DataFrame, 
                       factor_names: List[str] = None, 
                       return_periods: List[int] = None) -> pd.DataFrame:
    """
    根据IC值调整因子符号，使其与收益率正相关
    
    Args:
        df: 包含因子的DataFrame数据
        factor_names: 要调整的因子列表，如果为None则调整所有可用因子
        return_periods: 用于计算IC的收益率周期，默认为[10]
        
    Returns:
        调整符号后的DataFrame
    """
    register_all_factors()
    result_df = df.copy()
    
    # 如果未指定因子，则获取所有已计算的因子
    if factor_names is None:
        factor_info = FactorManager.get_factor_info()
        all_factors = factor_info['name'].tolist()
        factor_names = [col for col in df.columns if col in all_factors]
    
    # 使用默认收益率周期
    if return_periods is None:
        return_periods = [10]
        
    print(f"\n开始调整 {len(factor_names)} 个因子的符号...")
    config = FactorTestConfig(return_periods=return_periods)
    
    # 首先计算前向收益率
    result_df = FactorsTester.calculate_forward_returns(result_df, periods=return_periods)
    
    adjusted_factors = []
    for factor_name in factor_names:
        if factor_name not in result_df.columns:
            print(f"警告: 因子 {factor_name} 不在数据中，跳过")
            continue
            
        try:
            # 计算IC
            ic_df = FactorsTester.calculate_ic(
                result_df, 
                factor_names=factor_name,
                return_periods=return_periods, 
                method=config.ic_method
            )
            
            # 获取平均IC
            mean_ic = ic_df[f'{factor_name}_{return_periods[0]}period_ic'].mean()
            
            # 如果IC为负，调整因子符号
            if mean_ic < 0:
                result_df[factor_name] = -result_df[factor_name]
                print(f"调整因子 {factor_name} 的符号 (IC: {mean_ic:.4f})")
                adjusted_factors.append(factor_name)
                
        except Exception as e:
            print(f"调整因子 {factor_name} 符号时出错: {str(e)}")
            continue
    
    print(f"\n调整完成! 共调整了 {len(adjusted_factors)} 个因子的符号")
    if adjusted_factors:
        print("调整的因子列表:")
        print(", ".join(adjusted_factors))
        
    return result_df

def compute_and_save_by_date(data_dir: str,
                            start_date: Union[str, datetime.datetime],
                            end_date: Union[str, datetime.datetime],
                            factor_names: List[str],
                            output_dir: str = "factor_data_by_date",
                            chunk_size: int = 500000,
                            overlap_size: int = 100000,
                            adjust_sign: bool = True,
                            return_periods: List[int] = None) -> None:
    """
    按日期计算因子并分别保存，解决内存问题
    
    Args:
        data_dir: 原始数据所在目录
        start_date: 开始日期
        end_date: 结束日期
        factor_names: 要计算的因子列表
        output_dir: 输出目录，将按日期保存子文件
        chunk_size: 计算因子时的分块大小
        overlap_size: 计算因子时的重叠区域大小
        adjust_sign: 是否调整因子符号
        return_periods: 收益率周期列表，用于计算IC和调整符号
    """
    # 注册所有因子
    register_all_factors()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换日期
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 生成日期列表
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_dates = [date for date in date_range if date.weekday() < 5]  # 简单过滤非交易日
    
    print(f"将处理从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据")
    print(f"预计处理 {len(trading_dates)} 个交易日")
    
    # 初始化数据处理器
    data_processor = FinDataProcessor(data_dir)
    
    # 加载因子管理器获取因子信息
    factor_info = FactorManager.get_factor_info()
    available_factors = [f for f in factor_names if f in factor_info['name'].tolist()]
    
    if not available_factors:
        print("错误：指定的因子都不可用")
        return
    
    print(f"将计算以下因子: {available_factors}")
    
    # 设置默认的收益率周期
    if return_periods is None:
        return_periods = [10]
    
    # 逐个日期处理
    processed_dates = []
    for current_date in trading_dates:
        date_str = current_date.strftime('%Y%m%d')
        output_file = os.path.join(output_dir, f"factors_{date_str}.feather")
        
        # 如果文件已存在且不强制重新计算，则跳过
        if os.path.exists(output_file):
            print(f"日期 {date_str} 的因子数据已存在，跳过计算")
            processed_dates.append(date_str)
            continue
        
        try:
            print(f"\n处理日期: {date_str}")
            
            # 加载当日数据
            daily_file = os.path.join(data_dir, f"{date_str}.feather")
            if not os.path.exists(daily_file):
                print(f"日期 {date_str} 的数据文件不存在，跳过")
                continue
            
            try:
                df = pd.read_feather(daily_file)
                print(f"加载了 {len(df)} 行数据")
            except Exception as e:
                print(f"读取文件 {daily_file} 时出错: {str(e)}")
                continue
            
            if df.empty:
                print(f"日期 {date_str} 的数据为空，跳过")
                continue
            
            # 计算因子
            print(f"计算因子...")
            df_with_factors = compute_multiple_factors(df, factor_names, adjust_sign=False)
            
            # 计算收益率（如果需要）
            if adjust_sign or any(f"{p}period_return" not in df_with_factors.columns for p in return_periods):
                print(f"计算收益率...")
                
                # 确保TradDay列存在
                if 'TradDay' not in df_with_factors.columns and 'DateTime' in df_with_factors.columns:
                    df_with_factors['TradDay'] = df_with_factors['DateTime'].dt.date
                
                # 计算收益率
                for period in return_periods:
                    col_name = f'{period}period_return'
                    if col_name not in df_with_factors.columns:
                        df_with_factors[col_name] = df_with_factors.groupby('InstruID')['mid_price'].transform(
                            lambda x: x.pct_change(period).shift(-period)
                        )
            
            # 如果需要调整符号
            if adjust_sign:
                print(f"调整因子符号...")
                for factor in available_factors:
                    # 计算IC
                    for period in return_periods:
                        return_col = f'{period}period_return'
                        if return_col in df_with_factors.columns and factor in df_with_factors.columns:
                            # 选择有效数据
                            valid_data = df_with_factors[[factor, return_col]].dropna()
                            if len(valid_data) >= 30:  # 最小样本量
                                # 使用spearman相关系数
                                ic = stats.spearmanr(valid_data[factor], valid_data[return_col])[0]
                                if not np.isnan(ic) and ic < 0:
                                    print(f"  调整因子 {factor} 的符号 (IC={ic:.4f})")
                                    df_with_factors[factor] = -df_with_factors[factor]
                                    break  # 一旦调整就退出循环
            
            # 保存结果
            print(f"保存结果到: {output_file}")
            df_with_factors.to_feather(output_file)
            processed_dates.append(date_str)
            
            # 释放内存
            del df, df_with_factors
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"处理日期 {date_str} 时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # 保存处理成功的日期列表
    dates_file = os.path.join(output_dir, "processed_dates.txt")
    with open(dates_file, 'w') as f:
        for date in processed_dates:
            f.write(f"{date}\n")
    
    print(f"\n按日期计算因子完成!")
    print(f"成功处理了 {len(processed_dates)}/{len(trading_dates)} 个交易日")
    print(f"结果保存在目录: {output_dir}")
    print(f"处理成功的日期列表保存在: {dates_file}")

def load_factors_by_date_range(factors_dir: str,
                              start_date: Union[str, datetime.datetime] = None,
                              end_date: Union[str, datetime.datetime] = None,
                              factor_names: List[str] = None) -> pd.DataFrame:
    """
    加载指定日期范围内的因子数据
    
    Args:
        factors_dir: 因子数据目录，包含按日期保存的feather文件
        start_date: 开始日期，如果为None则从最早的日期开始
        end_date: 结束日期，如果为None则到最晚的日期结束
        factor_names: 要加载的因子列表，如果为None则加载所有因子
        
    Returns:
        合并后的DataFrame
    """
    # 获取所有可用的日期文件
    files = os.listdir(factors_dir)
    factor_files = [f for f in files if f.startswith("factors_") and f.endswith(".feather")]
    
    if not factor_files:
        raise ValueError(f"目录 {factors_dir} 中没有找到因子数据文件")
    
    # 转换日期
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
    
    # 解析文件名中的日期
    file_dates = []
    for file in factor_files:
        try:
            date_str = file[8:16]  # 提取文件名中的日期部分 (factors_YYYYMMDD.feather)
            date = pd.to_datetime(date_str)
            file_dates.append((date, file))
        except:
            continue
    
    # 按日期排序
    file_dates.sort(key=lambda x: x[0])
    
    # 筛选日期范围
    if start_date is not None:
        file_dates = [(date, file) for date, file in file_dates if date >= start_date]
    
    if end_date is not None:
        file_dates = [(date, file) for date, file in file_dates if date <= end_date]
    
    if not file_dates:
        raise ValueError(f"在指定的日期范围内没有找到因子数据文件")
    
    # 加载并合并数据
    all_data = []
    total_files = len(file_dates)
    
    print(f"将加载 {total_files} 个日期的因子数据")
    
    for i, (date, file) in enumerate(file_dates):
        try:
            print(f"加载 {i+1}/{total_files}: {date.strftime('%Y-%m-%d')} ({file})")
            filepath = os.path.join(factors_dir, file)
            df = pd.read_feather(filepath)
            
            # 如果指定了因子名称，则只保留需要的列
            if factor_names is not None:
                # 确保保留基本列
                base_cols = ['InstruID', 'DateTime', 'TradDay', 'date']
                keep_cols = [col for col in base_cols if col in df.columns]
                # 添加指定的因子列
                for factor in factor_names:
                    if factor in df.columns:
                        keep_cols.append(factor)
                # 添加收益率列
                return_cols = [col for col in df.columns if 'period_return' in col]
                keep_cols.extend(return_cols)
                
                # 只保留需要的列
                df = df[list(set(keep_cols))]
            
            all_data.append(df)
            
            # 释放内存
            del df
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"加载文件 {file} 时出错: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("所有文件加载失败")
    
    # 合并所有数据
    print("合并所有数据...")
    result = pd.concat(all_data, ignore_index=True)
    
    # 清理内存
    del all_data
    gc.collect()
    
    # 确保数据按时间排序
    if 'DateTime' in result.columns:
        result = result.sort_values(['DateTime', 'InstruID'])
    elif 'TradDay' in result.columns:
        result = result.sort_values(['TradDay', 'InstruID'])
    elif 'date' in result.columns:
        result = result.sort_values(['date', 'InstruID'])
    
    print(f"成功加载了 {len(result)} 行数据")
    
    return result

def create_batch_generator(factors_dir: str,
                          batch_size: int = 5,
                          factor_names: List[str] = None,
                          return_col: str = '10period_return') -> Iterator[pd.DataFrame]:
    """
    创建按批次加载因子数据的生成器
    
    Args:
        factors_dir: 因子数据目录，包含按日期保存的feather文件
        batch_size: 每批加载的日期数量
        factor_names: 要加载的因子列表，如果为None则加载所有因子
        return_col: 收益率列名
        
    Returns:
        生成器，每次返回一批数据
    """
    # 获取并排序所有处理过的日期
    try:
        with open(os.path.join(factors_dir, "processed_dates.txt"), 'r') as f:
            dates = [line.strip() for line in f.readlines()]
    except:
        # 如果没有日期列表文件，则从目录中获取所有日期
        files = os.listdir(factors_dir)
        dates = []
        for file in files:
            if file.startswith("factors_") and file.endswith(".feather"):
                try:
                    date_str = file[8:16]  # 提取文件名中的日期部分
                    dates.append(date_str)
                except:
                    continue
    
    # 按日期排序
    dates.sort()
    
    # 分批处理
    total_dates = len(dates)
    batches = (total_dates + batch_size - 1) // batch_size
    
    print(f"共有 {total_dates} 个日期，将分成 {batches} 批处理")
    
    for i in range(batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_dates)
        batch_dates = dates[start_idx:end_idx]
        
        print(f"加载第 {i+1}/{batches} 批，包含日期: {batch_dates}")
        
        # 加载这一批的数据
        batch_data = []
        for date_str in batch_dates:
            try:
                file_path = os.path.join(factors_dir, f"factors_{date_str}.feather")
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue
                
                df = pd.read_feather(file_path)
                
                # 如果指定了因子名称，则只保留需要的列
                if factor_names is not None:
                    # 确保保留基本列
                    base_cols = ['InstruID', 'DateTime', 'TradDay', 'date']
                    keep_cols = [col for col in base_cols if col in df.columns]
                    # 添加指定的因子列
                    for factor in factor_names:
                        if factor in df.columns:
                            keep_cols.append(factor)
                    # 添加指定的收益率列
                    if return_col in df.columns:
                        keep_cols.append(return_col)
                    
                    # 只保留需要的列
                    df = df[list(set(keep_cols))]
                
                batch_data.append(df)
                
            except Exception as e:
                print(f"加载日期 {date_str} 数据时出错: {str(e)}")
                continue
        
        if batch_data:
            # 合并批次数据
            batch_df = pd.concat(batch_data, ignore_index=True)
            
            # 清理内存
            del batch_data
            import gc
            gc.collect()
            
            # 确保结果中包含必要的列
            required_cols = factor_names if factor_names else []
            if return_col:
                required_cols.append(return_col)
            
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            if missing_cols:
                print(f"警告: 批次数据中缺少以下列: {missing_cols}")
            
            # 返回这一批数据
            yield batch_df
        else:
            print(f"第 {i+1} 批没有有效数据，跳过")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算因子并按日期保存')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--output-dir', type=str, default='factor_data_by_date', help='输出目录')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default='2022-12-31', help='结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--factor', type=str, default=None, help='要计算的因子名称，如果不指定则计算所有因子')
    args = parser.parse_args()
    
    # 注册所有因子
    register_all_factors()
    
    # 确定要计算的因子
    if args.factor:
        factor_names = [args.factor]
    else:
        # 获取所有因子
        factor_info = FactorManager.get_factor_info()
        factor_names = factor_info['name'].tolist()
    
    # 按日期计算并保存因子
    compute_and_save_by_date(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        factor_names=factor_names,
        output_dir=args.output_dir
    )
    
    print("\n示例：如何加载计算好的因子数据")
    print("方法1：加载指定日期范围的所有数据")
    print("    df = load_factors_by_date_range('factor_data_by_date', '2022-01-01', '2022-01-31')")
    
    print("\n方法2：使用批次生成器逐批加载数据")
    print("    for batch_df in create_batch_generator('factor_data_by_date', batch_size=5):")
    print("        # 处理每一批数据")
    print("        process_batch(batch_df)")