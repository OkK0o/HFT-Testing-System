# factor_example.py

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from fin_data_processor import FinDataProcessor
from factors_test import FactorsTester, FactorTestConfig
import pandas as pd
import datetime
from typing import List, Dict
import os

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