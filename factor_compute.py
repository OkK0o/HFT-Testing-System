# factor_example.py

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from fin_data_processor import FinDataProcessor
from factors_test import FactorsTester, FactorTestConfig
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Union, Tuple
import os
import gc

def run_factor_compute(df: pd.DataFrame, 
                      factor_name: str = None, 
                      config: FactorTestConfig = None,
                      save_path: str = "factor_test_results",
                      return_periods: List[int] = None,
                      generate_images: bool = False,
                      adjust_sign: bool = True) -> Dict:
    """
    运行因子测试
    所有因子（包括分钟级别的）都直接在原始数据上计算，不进行重采样
    
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
            # 所有因子都直接在原始数据上计算，无论频率
            # 但保留频率信息以便让FactorManager知道要计算的是哪种类型的因子
            factor_df = FactorManager.calculate_factors(df, frequency=factor_freq, factor_names=factor_name)
            
            if factor_name in factor_df.columns:
                data[factor_name] = factor_df[factor_name].values
                print(f"因子 {factor_name} 计算完成")
            else:
                print(f"警告: 因子 {factor_name} 未被成功计算")
            
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
    不再区分tick级别和分钟级别因子，所有因子都直接在原始数据上计算
    
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
        
    # 保存原始列，用于后续验证是否添加了新列
    original_columns = set(df.columns)
    original_index = df.index.copy()
    result_df = df.copy()
    
    try:
        # 所有因子都直接在原始数据上计算，无论频率
        # 但保留频率信息以便让FactorManager知道要计算的是哪种类型的因子
        factor_df = FactorManager.calculate_factors(df, frequency=factor_freq, factor_names=factor_name)
        
        # 检查因子是否计算成功
        if factor_name in factor_df.columns:
            # 直接将因子列赋值给结果DataFrame
            result_df[factor_name] = factor_df[factor_name].values
            print(f"因子 {factor_name} 计算完成")
        else:
            print(f"警告: 因子 {factor_name} 未被成功计算")
        
        # 恢复原始索引顺序
        result_df = result_df.loc[original_index]
        
        # 验证因子是否成功添加
        new_columns = set(result_df.columns)
        added_columns = new_columns - original_columns
        
        if factor_name in added_columns:
            print(f"成功添加因子: {factor_name}")
        elif factor_name in result_df.columns:
            print(f"因子列已存在: {factor_name}")
        else:
            print(f"警告: 计算后因子 {factor_name} 不在结果DataFrame中")
    except Exception as e:
        print(f"计算因子 {factor_name} 时出错: {str(e)}")
        return df
    
    return result_df

def compute_multiple_factors(df: pd.DataFrame, 
                           factor_names: List[str],
                           adjust_sign: bool = True,
                           use_period: int = None) -> pd.DataFrame:
    """
    批量计算多个因子
    
    Args:
        df: 输入的DataFrame数据
        factor_names: 要计算的因子名称列表
        adjust_sign: 是否根据IC调整因子符号
        use_period: 使用的return周期，如果提供将使用指定周期的return
        
    Returns:
        添加了所有因子值的原始DataFrame
    """
    result_df = df.copy()
    print(f"\n开始计算 {len(factor_names)} 个因子...")
    
    # 保存原始列，用于验证
    original_columns = set(result_df.columns)
    
    # 如果指定了use_period，确保相应的return列存在
    if use_period is not None:
        # 检查是否存在指定周期的return列
        return_col = f'{use_period}period_return'
        if return_col not in result_df.columns:
            print(f"警告: 未找到 {return_col} 列，将尝试使用其他可用的return")
            
        print(f"使用 {use_period} 周期的收益率计算因子")
    
    # 逐个计算因子
    successful_factors = []
    failed_factors = []
    
    for i, factor_name in enumerate(factor_names):
        print(f"\n计算因子 {i+1}/{len(factor_names)}: {factor_name}")
        
        # 计算单个因子
        result_df = compute_factor(result_df, factor_name)
        
        # 验证因子是否成功添加
        if factor_name in result_df.columns:
            successful_factors.append(factor_name)
            print(f"✓ 因子 {factor_name} 成功添加")
        else:
            failed_factors.append(factor_name)
            print(f"✗ 因子 {factor_name} 添加失败")
    
    # 如果需要调整符号
    if adjust_sign and successful_factors:
        print("\n调整因子符号...")
        # 如果指定了使用周期，使用该周期进行调整
        if use_period is not None:
            result_df = adjust_factor_signs(result_df, factor_names=successful_factors, return_periods=[use_period])
        else:
            result_df = adjust_factor_signs(result_df, factor_names=successful_factors)
    
    # 输出计算摘要
    print(f"\n因子计算完成摘要:")
    print(f"- 成功计算的因子: {len(successful_factors)}/{len(factor_names)}")
    print(f"- 失败的因子: {len(failed_factors)}/{len(factor_names)}")
    
    if failed_factors:
        print(f"失败的因子列表: {failed_factors}")
    
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
    
    # 获取因子信息和频率
    factor_info = FactorManager.get_factor_info()
    factor_freqs = {factor: FactorManager.get_factor_frequency(factor) for factor in factor_names}
    
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
        
        # 为当前块计算每个因子
        for factor_name in factor_names:
            try:
                # 获取因子频率
                factor_freq = factor_freqs.get(factor_name)
                if factor_freq is None:
                    print(f"警告: 因子 {factor_name} 的频率未知，跳过")
                    continue
                
                # 直接使用原始数据计算因子
                factor_df = FactorManager.calculate_factors(chunk_df, frequency=factor_freq, factor_names=factor_name)
                
                if factor_name in factor_df.columns:
                    # 直接将因子值添加到当前块
                    chunk_df[factor_name] = factor_df[factor_name].values
                    print(f"因子 {factor_name} 计算完成（块 {chunk_idx + 1}）")
                else:
                    print(f"警告: 因子 {factor_name} 未被成功计算（块 {chunk_idx + 1}）")
                    continue
                
                # 释放内存
                del factor_df
                gc.collect()
            except Exception as e:
                print(f"计算因子 {factor_name} 时出错 (块 {chunk_idx + 1}): {str(e)}")
                continue
        
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
            if factor in chunk_df.columns:
                # 取当前块的有效区域
                valid_chunk_indices = chunk_df.index[valid_start_idx:valid_end_idx]
                original_indices = df.index[start_idx + valid_start_idx:start_idx + valid_end_idx]
                
                # 将有效区域的因子值赋给最终结果
                if len(valid_chunk_indices) > 0:
                    # 确保索引匹配
                    factor_values[factor].loc[original_indices] = chunk_df.loc[valid_chunk_indices, factor].values
        
        # 释放内存
        del chunk_df
        gc.collect()
        print(f"第 {chunk_idx + 1} 块处理完成，已释放内存")
    
    # 将计算的因子添加到最终结果中
    for factor in factor_names:
        if not factor_values[factor].isna().all():  # 确保有计算结果
            final_df[factor] = factor_values[factor]
            print(f"因子 {factor} 添加完成，非空值: {factor_values[factor].count()}/{len(factor_values[factor])}")
        else:
            print(f"警告: 因子 {factor} 没有有效值")
    
    # 如果需要调整符号
    if adjust_sign:
        print("\n调整因子符号...")
        final_df = adjust_factor_signs_in_chunks(final_df, factor_names=factor_names, return_periods=return_periods)
    
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
    
    # 获取因子频率
    factor_freqs = {factor: FactorManager.get_factor_frequency(factor) for factor in factor_names}
    
    # 按合约处理数据
    for i, contract in enumerate(contracts):
        print(f"\n处理合约 {i+1}/{len(contracts)}: {contract}")
        
        # 提取当前合约的数据
        contract_df = df[df['InstruID'] == contract].copy()
        
        # 为当前合约计算所有因子
        for factor_name in factor_names:
            try:
                # 获取因子频率
                factor_freq = factor_freqs.get(factor_name)
                if factor_freq is None:
                    print(f"警告: 因子 {factor_name} 的频率未知，跳过")
                    continue
                
                print(f"计算因子: {factor_name} (合约: {contract})")
                
                # 直接使用原始数据计算因子
                factor_df = FactorManager.calculate_factors(contract_df, frequency=factor_freq, factor_names=factor_name)
                
                if factor_name in factor_df.columns:
                    # 将因子值添加到合约数据中
                    contract_df[factor_name] = factor_df[factor_name].values
                    print(f"因子 {factor_name} 计算完成 (合约: {contract})")
                else:
                    print(f"警告: 因子 {factor_name} 未被成功计算 (合约: {contract})")
                    continue
                
                # 将计算结果更新回结果数据框
                result_df.loc[contract_df.index, factor_name] = contract_df[factor_name].values
                
                # 释放内存
                del factor_df
                gc.collect()
            except Exception as e:
                print(f"计算因子 {factor_name} 时出错 (合约: {contract}): {str(e)}")
                continue
        
        # 释放内存
        del contract_df
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

# 添加因子平滑函数
def smooth_factor(df: pd.DataFrame, 
                 factor_name: str, 
                 window: int = 5, 
                 method: str = 'sma') -> pd.DataFrame:
    """
    平滑单个因子
    
    Args:
        df: 输入DataFrame，必须包含factor_name列
        factor_name: 要平滑的因子名称
        window: 平滑窗口大小
        method: 平滑方法，'sma'=简单移动平均，'ema'=指数移动平均
        
    Returns:
        添加了平滑因子的DataFrame
    """
    result_df = df.copy()
    
    # 检查因子是否存在
    if factor_name not in result_df.columns:
        raise ValueError(f"因子 '{factor_name}' 不在数据中")
    
    # 生成平滑因子名称
    method_suffix = 'sma' if method == 'sma' else 'ema'
    smoothed_name = f"{factor_name}_{method_suffix}{window}"
    
    # 确保数据已按时间排序
    if 'DateTime' in result_df.columns:
        result_df = result_df.sort_values(['InstruID', 'DateTime'])
    elif 'TradDay' in result_df.columns and 'UpdateTime' in result_df.columns:
        result_df = result_df.sort_values(['InstruID', 'TradDay', 'UpdateTime'])
    elif 'date' in result_df.columns:
        result_df = result_df.sort_values(['InstruID', 'date'])
    
    # 应用平滑
    if method == 'sma':
        # 简单移动平均
        result_df[smoothed_name] = result_df.groupby('InstruID')[factor_name].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    elif method == 'ema':
        # 指数移动平均
        result_df[smoothed_name] = result_df.groupby('InstruID')[factor_name].transform(
            lambda x: x.ewm(span=window, min_periods=1).mean()
        )
    else:
        raise ValueError(f"不支持的平滑方法: {method}，支持的方法有 'sma', 'ema'")
    
    print(f"平滑因子 {factor_name} 完成 -> {smoothed_name}")
    return result_df

def smooth_factors(df: pd.DataFrame, 
                  factor_names: List[str], 
                  windows: Union[List[int], int] = 5, 
                  methods: Union[List[str], str] = 'sma',
                  register_factors: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    批量平滑多个因子
    
    Args:
        df: 输入DataFrame，必须包含factor_names列
        factor_names: 要平滑的因子名称列表
        windows: 平滑窗口大小，可以是单个整数或与factor_names等长的列表
        methods: 平滑方法，可以是单个字符串或与factor_names等长的列表
        register_factors: 是否将平滑因子注册到FactorManager中
        
    Returns:
        (添加了平滑因子的DataFrame, 生成的平滑因子名称列表)
    """
    result_df = df.copy()
    smoothed_names = []
    
    # 规范化windows参数
    if isinstance(windows, int):
        windows = [windows] * len(factor_names)
    elif len(windows) != len(factor_names):
        raise ValueError("windows参数长度必须为1或与factor_names相同")
    
    # 规范化methods参数
    if isinstance(methods, str):
        methods = [methods] * len(factor_names)
    elif len(methods) != len(factor_names):
        raise ValueError("methods参数长度必须为1或与factor_names相同")
    
    # 逐个因子处理
    for i, factor_name in enumerate(factor_names):
        window = windows[i]
        method = methods[i]
        
        # 检查因子是否存在
        if factor_name not in result_df.columns:
            print(f"警告: 因子 '{factor_name}' 不在数据中，跳过")
            continue
        
        # 生成平滑因子名称
        method_suffix = 'sma' if method == 'sma' else 'ema'
        smoothed_name = f"{factor_name}_{method_suffix}{window}"
        
        # 应用平滑
        if method == 'sma':
            # 简单移动平均
            result_df[smoothed_name] = result_df.groupby('InstruID')[factor_name].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        elif method == 'ema':
            # 指数移动平均
            result_df[smoothed_name] = result_df.groupby('InstruID')[factor_name].transform(
                lambda x: x.ewm(span=window, min_periods=1).mean()
            )
        else:
            print(f"警告: 不支持的平滑方法: {method}，跳过因子 {factor_name}")
            continue
        
        # 注册平滑因子
        if register_factors:
            try:
                FactorManager.register_smoothed_factor(factor_name, window, method)
                print(f"因子 {smoothed_name} 已注册到FactorManager")
            except Exception as e:
                print(f"注册因子 {smoothed_name} 失败: {str(e)}")
        
        smoothed_names.append(smoothed_name)
        print(f"平滑因子 {factor_name} 完成 -> {smoothed_name}")
    
    print(f"批量平滑完成，共添加 {len(smoothed_names)} 个平滑因子")
    return result_df, smoothed_names

if __name__ == "__main__":
    df = pd.read_feather("data.feather")
    
    # 示例1：计算单个因子
    df_with_factor = compute_factor(df, factor_name="price_reversal")
    
    # 示例2：批量计算多个因子
    factors_to_compute = ['price_reversal', 'kyle_lambda', 'unit_return_volume']
    df_with_factors = compute_multiple_factors(df, factors_to_compute)
    
    # 示例3：计算并测试因子
    results = run_factor_compute(df, factor_name="price_reversal")
    
    # 示例4：使用自定义配置测试因子
    custom_config = FactorTestConfig(
        ic_method='pearson',
        ic_threshold=0.03,
        ir_threshold=0.6
    )
    results = run_factor_compute(df, factor_name="weighted_momentum_10t", config=custom_config, return_periods=[100])
    
    # 示例5：显示所有因子
    run_factor_compute(df)