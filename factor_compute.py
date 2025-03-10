# factor_example.py

from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from fin_data_processor import FinDataProcessor
from factors_test import FactorsTester, FactorTestConfig
import pandas as pd
import datetime
from typing import List

def run_factor_compute(df: pd.DataFrame, 
                      factor_name: str = None, 
                      config: FactorTestConfig = None,
                      save_path: str = "factor_test_results",
                      return_periods: List[int] = None):
    """
    运行因子测试
    
    Args:
        df: 输入的DataFrame数据
        factor_name: 要测试的因子名称，如果为None则显示所有因子
        config: 测试配置，如果为None则使用默认配置
        save_path: 结果保存路径
        return_periods: 自定义收益率计算周期列表，如[1, 5, 10, 20]
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
        results = tester.test_single_factor(factor_name, data, config)
        return results
    else:
        print("\n=== 所有已注册因子 ===")
        all_factors = FactorManager.get_factor_info()
        print(all_factors)
        return None

def compute_factor(df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
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

def compute_multiple_factors(df: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
    """
    批量计算多个因子
    
    Args:
        df: 输入的DataFrame数据
        factor_names: 要计算的因子名称列表
        
    Returns:
        添加了所有因子值的原始DataFrame
    """
    result_df = df.copy()
    print(f"\n开始计算 {len(factor_names)} 个因子...")
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