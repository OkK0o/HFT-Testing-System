import pandas as pd
import numpy as np
from typing import Union, List, Dict, Callable, Set, Optional, Tuple
import warnings
from scipy import stats
from functools import wraps
from enum import Enum

class FactorFrequency(Enum):
    """因子频率枚举"""
    TICK = "tick"
    MINUTE = "minute"
    SMOOTHED = "smoothed"  # 添加平滑因子类型

class FactorRegistry:
    """因子注册管理器"""
    
    def __init__(self):
        self.factors: Dict[str, Dict] = {}  # 存储因子信息
        self.categories: Dict[str, Set[str]] = {}  # 因子分类
        self.dependencies: Dict[str, Set[str]] = {}  # 因子依赖关系
        self.frequencies: Dict[str, FactorFrequency] = {}  # 因子频率
        self.smoothed_factors: Dict[str, Dict] = {}  # 存储平滑因子信息
    
    def register(self, 
                name: str, 
                frequency: FactorFrequency,
                category: str = 'other',
                description: str = '',
                dependencies: List[str] = None) -> Callable:
        """
        因子注册装饰器
        
        Args:
            name: 因子名称
            frequency: 因子频率（TICK/MINUTE）
            category: 因子类别
            description: 因子描述
            dependencies: 依赖的其他因子
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            self.factors[name] = {
                'function': func,
                'category': category,
                'description': description,
                'dependencies': dependencies or [],
                'frequency': frequency
            }
            
            # 更新分类
            if category not in self.categories:
                self.categories[category] = set()
            self.categories[category].add(name)
            
            if dependencies:
                self.dependencies[name] = set(dependencies)
            
            self.frequencies[name] = frequency
            
            return wrapper
        return decorator
        
    def register_smoothed_factor(self, 
                               original_name: str, 
                               window: int,
                               method: str = 'sma') -> str:
        """
        注册平滑因子
        
        Args:
            original_name: 原始因子名称
            window: 平滑窗口大小
            method: 平滑方法，'sma'=简单移动平均，'ema'=指数移动平均
            
        Returns:
            平滑因子名称
        """
        if original_name not in self.factors:
            raise ValueError(f"原始因子 '{original_name}' 不存在")
            
        # 生成平滑因子名称
        method_suffix = 'sma' if method == 'sma' else 'ema'
        smoothed_name = f"{original_name}_{method_suffix}{window}"
        
        # 原始因子的信息
        original_info = self.factors[original_name]
        
        # 注册平滑因子
        self.factors[smoothed_name] = {
            'function': None,  # 平滑因子的函数是动态计算的，不需要提前定义
            'category': 'smoothed',  # 使用专门的平滑因子类别
            'description': f"{original_info['description']} ({method_suffix.upper()}{window}平滑)",
            'dependencies': [original_name],  # 依赖原始因子
            'frequency': FactorFrequency.SMOOTHED,  # 使用SMOOTHED频率类型
            'smoothing_info': {
                'original_factor': original_name,
                'window': window,
                'method': method
            }
        }
        
        # 更新分类
        if 'smoothed' not in self.categories:
            self.categories['smoothed'] = set()
        self.categories['smoothed'].add(smoothed_name)
        
        # 更新依赖
        self.dependencies[smoothed_name] = set([original_name])
        
        # 更新频率
        self.frequencies[smoothed_name] = FactorFrequency.SMOOTHED
        
        # 记录平滑因子信息
        self.smoothed_factors[smoothed_name] = {
            'original_factor': original_name,
            'window': window,
            'method': method
        }
        
        return smoothed_name
    
    def get_factor_info(self, frequency: Optional[FactorFrequency] = None) -> pd.DataFrame:
        """
        获取因子信息
        
        Args:
            frequency: 可选的频率过滤
            
        Returns:
            因子信息DataFrame
        """
        records = []
        for name, info in self.factors.items():
            if frequency and info['frequency'].value != frequency.value:
                continue
                
            # 添加平滑信息(如果是平滑因子)
            smoothing_info = ""
            if 'smoothing_info' in info:
                smooth_data = info['smoothing_info']
                smoothing_info = f"平滑自: {smooth_data['original_factor']}, 窗口: {smooth_data['window']}, 方法: {smooth_data['method']}"
                
            records.append({
                'name': name,
                'frequency': info['frequency'].value,
                'category': info['category'],
                'description': info['description'],
                'dependencies': ', '.join(info['dependencies'] or []),
                'smoothing_info': smoothing_info
            })
        return pd.DataFrame(records)
    
    def get_factors_by_category(self, 
                              category: str,
                              frequency: Optional[FactorFrequency] = None) -> List[str]:
        """
        获取指定类别的因子
        
        Args:
            category: 因子类别
            frequency: 可选的频率过滤
            
        Returns:
            因子名称列表
        """
        factors = list(self.categories.get(category, set()))
        if frequency:
            factors = [f for f in factors if self.frequencies[f].value == frequency.value]
        return factors

class FactorManager:
    """因子管理器"""
    
    registry = FactorRegistry()
    
    @staticmethod
    def get_factor_names(frequency: Optional[FactorFrequency] = None) -> List[str]:
        """获取因子名称"""
        if frequency:
            return [name for name, freq in FactorManager.registry.frequencies.items() 
                   if freq.value == frequency.value]
        return list(FactorManager.registry.factors.keys())
    
    @staticmethod
    def get_factor_info(frequency: Optional[FactorFrequency] = None) -> pd.DataFrame:
        """获取因子信息"""
        return FactorManager.registry.get_factor_info(frequency)
    
    @staticmethod
    def get_factor_frequency(factor_name: str) -> Optional[FactorFrequency]:
        """
        获取因子的频率
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子频率，如果因子不存在则返回None
        """
        return FactorManager.registry.frequencies.get(factor_name)
    
    @staticmethod
    def calculate_factors(df: pd.DataFrame,
                         frequency: FactorFrequency,
                         factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算因子
        
        Args:
            df: 输入数据
            frequency: 因子频率
            factor_names: 要计算的因子名称列表，如果为None则显示所有可用因子
            
        Returns:
            计算结果DataFrame
        """
        try:
            if df is None:
                raise ValueError("输入数据为None")
                
            result = df.copy()
            
            if factor_names is None:
                print("\n请指定要计算的因子名称")
                print("\n可用的因子列表：")
                print(FactorManager.get_factor_info())
                return df
            elif isinstance(factor_names, str):
                factor_names = [factor_names]
            for name in factor_names:
                if name not in FactorManager.registry.factors:
                    raise ValueError(f"因子 '{name}' 不存在")
                
            dependencies = FactorManager.registry.dependencies
            
            factors_to_calculate = set(factor_names)
            for factor in factor_names:
                if factor in dependencies:
                    factors_to_calculate.update(dependencies[factor])
            
            print(f"\n开始计算因子: {', '.join(factors_to_calculate)}")
            print(f"数据列名: {list(result.columns)}")
            print(f"数据形状: {result.shape}")
            
            calculated = set()
            while len(calculated) < len(factors_to_calculate):
                for name in factors_to_calculate:
                    if name in calculated:
                        continue
                    
                    deps = dependencies.get(name, set())
                    if deps and not deps.issubset(calculated):
                        continue
                    
                    print(f"\n计算因子: {name}")
                    info = FactorManager.registry.factors[name]
                    
                    # 检查是否是平滑因子
                    if info['frequency'] == FactorFrequency.SMOOTHED and 'smoothing_info' in info:
                        smooth_info = info['smoothing_info']
                        original_factor = smooth_info['original_factor']
                        window = smooth_info['window']
                        method = smooth_info['method']
                        
                        # 确保原始因子已计算
                        if original_factor not in result.columns:
                            raise ValueError(f"原始因子 '{original_factor}' 尚未计算")
                        
                        # 应用平滑
                        if method == 'sma':
                            result[name] = result.groupby('InstruID')[original_factor].transform(
                                lambda x: x.rolling(window=window, min_periods=1).mean()
                            )
                        elif method == 'ema':
                            result[name] = result.groupby('InstruID')[original_factor].transform(
                                lambda x: x.ewm(span=window, min_periods=1).mean()
                            )
                        else:
                            raise ValueError(f"不支持的平滑方法: {method}")
                    else:
                        # 常规因子计算
                        if info['function'] is not None:
                            result = info['function'](result)
                    
                    calculated.add(name)
            
            print(f"\n因子计算完成!")
            return result
            
        except Exception as e:
            print(f"\n错误发生在calculate_factors中:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            raise
    
    @staticmethod
    def register_smoothed_factor(original_name: str, window: int, method: str = 'sma') -> str:
        """
        注册平滑因子
        
        Args:
            original_name: 原始因子名称
            window: 平滑窗口大小
            method: 平滑方法，'sma'=简单移动平均，'ema'=指数移动平均
            
        Returns:
            平滑因子名称
        """
        return FactorManager.registry.register_smoothed_factor(original_name, window, method)

def register_example_factors():
    """注册示例因子"""
    
    @FactorManager.registry.register(
        name="minute_momentum",
        frequency=FactorFrequency.MINUTE,
        category="momentum",
        description="分钟级动量因子"
    )
    def calculate_minute_momentum(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        result = df.copy()
        result['minute_momentum'] = result.groupby('InstruID')['LastPrice'].pct_change(window)
        return result
    
    @FactorManager.registry.register(
        name="tick_order_flow",
        frequency=FactorFrequency.TICK,
        category="flow",
        description="Tick级订单流因子"
    )
    def calculate_tick_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['trade_direction'] = 0
        mask_buy = result['LastPrice'] >= result['AskPrice1']
        mask_sell = result['LastPrice'] <= result['BidPrice1']
        result.loc[mask_buy, 'trade_direction'] = 1
        result.loc[mask_sell, 'trade_direction'] = -1
        result['order_flow'] = result['trade_direction'] * result['Volume']
        return result
    

# 使用示例
if __name__ == "__main__":
    # 注册示例因子
    register_example_factors()
    
    print("\n所有因子信息:")
    print(FactorManager.get_factor_info())
    

    print("\n动量类因子:")
    print(FactorManager.registry.get_factors_by_category('momentum'))