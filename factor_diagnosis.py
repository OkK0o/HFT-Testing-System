import pandas as pd
import numpy as np
from typing import Dict, List
from factor_register import FactorRegister
from factor_manager import FactorManager, FactorFrequency

def diagnose_toxicity_factor(df: pd.DataFrame) -> Dict:
    """
    诊断 order_flow_toxicity 因子计算过程中的样本量变化
    
    Args:
        df: 输入的DataFrame
        
    Returns:
        包含诊断信息的字典
    """
    result = df.copy()
    diagnostics = {}
    
    # 1. 基础数据统计
    diagnostics['total_rows'] = len(df)
    diagnostics['contracts'] = df['InstruID'].nunique()
    diagnostics['trading_days'] = df['TradDay'].nunique()
    
    # 2. 检查必要列的存在性和有效性
    required_cols = ['mid_price', 'AskPrice1', 'BidPrice1', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    diagnostics['missing_columns'] = missing_cols
    
    if 'mid_price' not in df.columns:
        result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
    
    # 3. 检查各列的缺失值
    null_stats = {}
    for col in required_cols:
        if col in result.columns:
            null_stats[col] = {
                'null_count': result[col].isnull().sum(),
                'null_percentage': (result[col].isnull().sum() / len(result) * 100)
            }
    diagnostics['null_statistics'] = null_stats
    
    # 4. 检查异常值
    invalid_stats = {}
    if 'AskPrice1' in result.columns:
        invalid_stats['zero_ask'] = (result['AskPrice1'] == 0).sum()
    if 'BidPrice1' in result.columns:
        invalid_stats['zero_bid'] = (result['BidPrice1'] == 0).sum()
    if 'Volume' in result.columns:
        invalid_stats['zero_volume'] = (result['Volume'] == 0).sum()
    diagnostics['invalid_statistics'] = invalid_stats
    
    # 5. 检查交易方向判断
    result['trade_direction'] = 0
    result['LastPrice'] = result['LastPrice'].fillna(result['mid_price'])  # 如果没有LastPrice就用mid_price
    
    # 如果成交价大于等于卖一价，说明是主动买入
    result.loc[result['LastPrice'] >= result['AskPrice1'], 'trade_direction'] = 1
    # 如果成交价小于等于买一价，说明是主动卖出
    result.loc[result['LastPrice'] <= result['BidPrice1'], 'trade_direction'] = -1
    
    direction_stats = {
        'buy': (result['trade_direction'] == 1).sum(),
        'sell': (result['trade_direction'] == -1).sum(),
        'neutral': (result['trade_direction'] == 0).sum()
    }
    diagnostics['direction_statistics'] = direction_stats
    
    # 6. 检查滚动窗口计算
    result['signed_volume'] = result['Volume'] * result['trade_direction']
    
    window_sizes = [50, 100, 200]  # 测试不同窗口大小
    window_stats = {}
    
    for window in window_sizes:
        min_periods = window // 2
        toxicity = result.groupby('InstruID')['signed_volume'].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).sum()
        ) / (result.groupby('InstruID')['Volume'].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).sum()
        ) + 1e-9)
        
        window_stats[window] = {
            'non_null_count': toxicity.notna().sum(),
            'valid_percentage': (toxicity.notna().sum() / len(toxicity) * 100)
        }
    
    diagnostics['window_statistics'] = window_stats
    
    # 7. 按合约统计
    contract_stats = {}
    for contract in result['InstruID'].unique():
        contract_data = result[result['InstruID'] == contract]
        contract_stats[contract] = {
            'total_rows': len(contract_data),
            'valid_rows': contract_data[required_cols].notna().all(axis=1).sum(),
            'trading_days': contract_data['TradDay'].nunique()
        }
    diagnostics['contract_statistics'] = contract_stats
    
    return diagnostics

def print_diagnostics(diagnostics: Dict):
    """打印诊断结果"""
    print("\n=== 数据基础统计 ===")
    print(f"总行数: {diagnostics['total_rows']:,}")
    print(f"合约数: {diagnostics['contracts']}")
    print(f"交易日数: {diagnostics['trading_days']}")
    
    print("\n=== 缺失列检查 ===")
    if diagnostics['missing_columns']:
        print("缺失的必要列:", diagnostics['missing_columns'])
    else:
        print("所有必要列都存在")
    
    print("\n=== 缺失值统计 ===")
    for col, stats in diagnostics['null_statistics'].items():
        print(f"{col}:")
        print(f"  - 缺失值数量: {stats['null_count']:,}")
        print(f"  - 缺失值比例: {stats['null_percentage']:.2f}%")
    
    print("\n=== 异常值统计 ===")
    for key, value in diagnostics['invalid_statistics'].items():
        print(f"{key}: {value:,}")
    
    print("\n=== 交易方向统计 ===")
    total = sum(diagnostics['direction_statistics'].values())
    for direction, count in diagnostics['direction_statistics'].items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{direction}: {count:,} ({percentage:.2f}%)")
    
    print("\n=== 滚动窗口统计 ===")
    for window, stats in diagnostics['window_statistics'].items():
        print(f"\n窗口大小 {window}:")
        print(f"  - 有效值数量: {stats['non_null_count']:,}")
        print(f"  - 有效值比例: {stats['valid_percentage']:.2f}%")
    
    print("\n=== 合约统计 ===")
    for contract, stats in diagnostics['contract_statistics'].items():
        valid_percentage = (stats['valid_rows'] / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
        print(f"\n{contract}:")
        print(f"  - 总行数: {stats['total_rows']:,}")
        print(f"  - 有效行数: {stats['valid_rows']:,} ({valid_percentage:.2f}%)")
        print(f"  - 交易日数: {stats['trading_days']}")

def main():
    """主函数"""
    # 加载数据
    print("正在加载数据...")
    df = pd.read_feather("data.feather")
    
    # 运行诊断
    print("正在进行诊断...")
    diagnostics = diagnose_toxicity_factor(df)
    
    # 打印诊断结果
    print_diagnostics(diagnostics)

if __name__ == "__main__":
    main() 