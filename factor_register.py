# factor_register.py

from factor_manager import FactorManager, FactorFrequency
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Callable

class FactorRegister:
    """因子注册器
    因子类别:
        'momentum': '动量类因子',
        'volatility': '波动率类因子', 
        'volume': '成交量类因子',
        'orderbook': '订单簿类因子',
        'microstructure': '市场微观结构类因子',
        'time': '时间特征类因子',
        'term_structure': '期限结构类因子',
        'composite': '复合类因子'
    """
    
    @staticmethod
    def register_momentum_factors():
        """动量类因子"""
        # Tick频率动量因子（使用较小窗口）
        tick_windows = [10, 20, 50, 100]
        for window in tick_windows:
            # 普通动量因子
            @FactorManager.registry.register(
                name=f"momentum_{window}",
                frequency=FactorFrequency.TICK,
                category="momentum",
                description=f"{window}条Tick动量因子"
            )
            def calculate_momentum(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                # 确保有mid_price列
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                result[f'momentum_{w}'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(w)
                )
                return result
            
            # 成交量加权动量因子
            @FactorManager.registry.register(
                name=f"weighted_momentum_{window}",
                frequency=FactorFrequency.TICK,
                category="momentum",
                description=f"{window}条Tick加权动量因子"
            )
            def calculate_weighted_momentum(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                # 确保有mid_price列
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                price_change = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(w)
                )
                volume_weight = result.groupby('InstruID')['Volume'].transform(
                    lambda x: x / x.rolling(window=w, min_periods=1).sum()
                )
                result[f'weighted_momentum_{w}'] = price_change * volume_weight
                return result
        
        # 分钟频率动量因子（使用较大窗口）
        minute_windows = [300, 600, 1200]  # 5分钟、10分钟、20分钟对应的数据条数
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"momentum_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}条数据动量因子"
            )
            def calculate_momentum(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                # 确保有mid_price列
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                result[f'momentum_{w}'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(w)
                )
                return result
    
    @staticmethod
    def register_volatility_factors():
        """波动率类因子"""
        # Tick频率波动率因子
        tick_windows = [50, 100, 200]
        for window in tick_windows:
            # 实现波动率
            @FactorManager.registry.register(
                name=f"realized_vol_{window}",
                frequency=FactorFrequency.TICK,
                category="volatility",
                description=f"{window}条Tick实现波动率"
            )
            def calculate_realized_vol(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                # 确保有mid_price列
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                returns = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change()
                )
                result[f'realized_vol_{w}'] = result.groupby('InstruID')[returns.name].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).std()
                )
                return result
            
            # 高低价波动率
            @FactorManager.registry.register(
                name=f"high_low_vol_{window}",
                frequency=FactorFrequency.TICK,
                category="volatility",
                description=f"{window}条Tick高低价波动率"
            )
            def calculate_high_low_vol(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                result['log_hl_ratio'] = np.log(result['HighPrice'] / result['LowPrice'])
                result[f'high_low_vol_{w}'] = result.groupby('InstruID')['log_hl_ratio'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).std()
                )
                return result
        
        # 分钟频率波动率因子
        minute_windows = [300, 600, 1200]
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"volatility_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volatility",
                description=f"{window}条数据波动率因子"
            )
            def calculate_volatility(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                result['log_returns'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: np.log(x / x.shift(1))
                )
                result[f'volatility_{w}'] = result.groupby('InstruID')['log_returns'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).std()
                )
                return result
    
    @staticmethod
    def register_volume_factors():
        """成交量类因子"""
        # Tick频率成交量因子
        @FactorManager.registry.register(
            name="volume_intensity_100",
            frequency=FactorFrequency.TICK,
            category="volume",
            description="100条Tick成交量强度因子"
        )
        def calculate_volume_intensity(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            result['volume_intensity_100'] = df['Volume'] / (
                df.groupby('InstruID')['Volume'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ) + 1e-9
            )
            return result
        
        # 分钟频率成交量因子
        minute_windows = [300, 600, 1200]
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"volume_momentum_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volume",
                description=f"{window}条数据成交量动量因子"
            )
            def calculate_volume_momentum(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                result[f'volume_momentum_{w}'] = result.groupby('InstruID')['Volume'].transform(
                    lambda x: x.pct_change(w)
                )
                return result
    
    @staticmethod
    def register_orderbook_factors():
        """订单簿类因子"""
        @FactorManager.registry.register(
            name="order_book_imbalance",
            frequency=FactorFrequency.TICK,
            category="orderbook",
            description="订单簿不平衡因子"
        )
        def calculate_order_book_imbalance(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
            result = df.copy()
            
            # 计算各档位的量价积
            ask_pressure = 0
            bid_pressure = 0
            
            for i in range(1, levels + 1):
                ask_price = f'AskPrice{i}'
                bid_price = f'BidPrice{i}'
                ask_vol = f'AskVolume{i}'
                bid_vol = f'BidVolume{i}'
                
                if all(col in df.columns for col in [ask_price, bid_price, ask_vol, bid_vol]):
                    ask_pressure += df[ask_price] * df[ask_vol] * (1 / i)
                    bid_pressure += df[bid_price] * df[bid_vol] * (1 / i)
            
            result['order_book_imbalance'] = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + 1e-9)
            return result
    
    @staticmethod
    def register_microstructure_factors():
        """市场微观结构类因子"""
        @FactorManager.registry.register(
            name="effective_spread",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="有效价差"
        )
        def calculate_effective_spread(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            mid_price = (result['AskPrice1'] + result['BidPrice1']) / 2
            result['effective_spread'] = 2 * abs(result['mid_price'] - mid_price) / mid_price
            return result
        
        @FactorManager.registry.register(
            name="amihud_illiquidity",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="Amihud非流动性因子"
        )
        def calculate_amihud(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            returns = result.groupby('InstruID')['mid_price'].transform(lambda x: abs(x.pct_change()))
            volume = result.groupby('InstruID')['Volume'].transform(lambda x: x * result['mid_price'])
            result['amihud_illiquidity'] = returns / (volume + 1e-9)
            result['amihud_illiquidity'] = result.groupby('InstruID')['amihud_illiquidity'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            return result

        @FactorManager.registry.register(
            name="order_flow_toxicity",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="订单流毒性指标"
        )
        def calculate_toxicity(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            """
            计算订单流毒性因子
            
            Args:
                df: 输入数据
                window: 滚动窗口大小
                
            Returns:
                添加了order_flow_toxicity的DataFrame
            """
            result = df.copy()
            
            # 1. 确保有mid_price
            if 'mid_price' not in result.columns:
                result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
            
            # 2. 判断交易方向
            result['trade_direction'] = 0
            result['LastPrice'] = result['LastPrice'].fillna(result['mid_price'])
            
            # 主动买入：成交价大于等于卖一价
            result.loc[result['LastPrice'] >= result['AskPrice1'], 'trade_direction'] = 1
            # 主动卖出：成交价小于等于买一价
            result.loc[result['LastPrice'] <= result['BidPrice1'], 'trade_direction'] = -1
            
            # 3. 计算带符号的成交量
            result['signed_volume'] = result['Volume'] * result['trade_direction']
            
            # 4. 计算毒性指标
            # 按合约和交易日分组计算
            result['order_flow_toxicity'] = result.groupby(['InstruID', 'TradDay'])['signed_volume'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()  # 降低min_periods要求
            ) / (result.groupby(['InstruID', 'TradDay'])['Volume'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()  # 降低min_periods要求
            ) + 1e-9)  # 避免除零
            
            return result

        @FactorManager.registry.register(
            name="volume_synchronized_probability",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="成交量同步概率"
        )
        def calculate_vsp(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            returns = result.groupby('InstruID')['mid_price'].transform(lambda x: x.pct_change())
            volume_change = result.groupby('InstruID')['Volume'].transform(lambda x: x.pct_change())
            
            result['volume_synchronized_probability'] = ((returns > 0) == (volume_change > 0)).astype(float)
            result['volume_synchronized_probability'] = result.groupby('InstruID')['volume_synchronized_probability'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            return result

        @FactorManager.registry.register(
            name="bid_ask_pressure",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="买卖压力比率"
        )
        def calculate_pressure(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
            result = df.copy()
            # 计算买卖压力
            ask_pressure = result['AskVolume1'] / result['AskPrice1']
            bid_pressure = result['BidVolume1'] * result['BidPrice1']
            result['bid_ask_pressure'] = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + 1e-9)
            result['bid_ask_pressure'] = result.groupby('InstruID')['bid_ask_pressure'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            return result

        @FactorManager.registry.register(
            name="price_impact",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="价格冲击因子"
        )
        def calculate_price_impact(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            price_change = result.groupby('InstruID')['mid_price'].transform(lambda x: abs(x.pct_change()))
            norm_volume = result.groupby('InstruID')['Volume'].transform(
                lambda x: x / x.rolling(window=window, min_periods=window//2).std()
            )
            result['price_impact'] = price_change / (norm_volume + 1e-9)
            result['price_impact'] = result.groupby('InstruID')['price_impact'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            return result

        @FactorManager.registry.register(
            name="quote_slope",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="报价斜率"
        )
        def calculate_quote_slope(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
            result = df.copy()
            quote_diff = (result['AskPrice1'] - result['BidPrice1']) / result['mid_price']
            volume_sum = result['AskVolume1'] + result['BidVolume1']
            result['quote_slope'] = quote_diff / (volume_sum + 1e-9)
            result['quote_slope'] = result.groupby('InstruID')['quote_slope'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            return result

        @FactorManager.registry.register(
            name="price_reversal",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="价格反转因子"
        )
        def calculate_price_reversal(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
            result = df.copy()
            result['short_return'] = result.groupby('InstruID')['mid_price'].transform(
                lambda x: x.pct_change(1)
            )
            result['price_reversal'] = -1 * result.groupby('InstruID')['short_return'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            return result

        @FactorManager.registry.register(
            name="vpin",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="成交量导向的概率知情交易指标"
        )
        def calculate_vpin(df: pd.DataFrame, window: int = 50, bucket_size: int = None) -> pd.DataFrame:
            """
            计算 VPIN (Volume-Synchronized Probability of Informed Trading) 因子
            
            Args:
                df: 输入数据
                window: 滚动窗口大小
                bucket_size: 成交量桶大小，如果为None则自动设置为日均成交量的1/50
                
            Returns:
                添加了vpin的DataFrame
            """
            result = df.copy()
            
            # 1. 确保有mid_price
            if 'mid_price' not in result.columns:
                result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
            
            # 2. 计算成交量桶大小（如果未指定）
            if bucket_size is None:
                bucket_size = int(result.groupby('TradDay')['Volume'].sum().mean() / 50)
            
            # 3. 计算买卖方向
            result['trade_direction'] = 0
            result['LastPrice'] = result['LastPrice'].fillna(result['mid_price'])
            result.loc[result['LastPrice'] >= result['AskPrice1'], 'trade_direction'] = 1
            result.loc[result['LastPrice'] <= result['BidPrice1'], 'trade_direction'] = -1
            
            # 4. 计算带符号的成交量
            result['signed_volume'] = result['Volume'] * result['trade_direction']
            
            # 5. 按交易日分组计算VPIN
            def calculate_daily_vpin(group):
                # 累计成交量
                group['cum_volume'] = group['Volume'].cumsum()
                # 计算桶编号
                group['bucket'] = (group['cum_volume'] / bucket_size).astype(int)
                # 计算每个桶的买卖成交量
                bucket_volumes = group.groupby('bucket').agg({
                    'signed_volume': 'sum',
                    'Volume': 'sum'
                })
                # 计算每个桶的买卖失衡比例
                bucket_volumes['vpin'] = abs(bucket_volumes['signed_volume']) / bucket_volumes['Volume']
                # 使用滚动窗口计算VPIN
                bucket_volumes['vpin'] = bucket_volumes['vpin'].rolling(
                    window=window, 
                    min_periods=1
                ).mean()
                # 将VPIN值映射回原始数据
                return group['bucket'].map(bucket_volumes['vpin'])
            
            # 按合约和交易日分组计算VPIN
            result['vpin'] = result.groupby(['InstruID', 'TradDay']).apply(
                calculate_daily_vpin
            ).reset_index(level=[0,1], drop=True)
            
            # 6. 填充缺失值
            result['vpin'] = result.groupby('InstruID')['vpin'].fillna(method='ffill')
            
            return result

        @FactorManager.registry.register(
            name="hft_trend",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="高频趋势因子"
        )
        def calculate_hft_trend(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            
            price_direction = result.groupby('InstruID')['mid_price'].transform(
                lambda x: np.sign(x - x.shift(1))
            )
            
            volume_std = result.groupby('InstruID')['Volume'].transform(
                lambda x: x / x.rolling(window=window, min_periods=window//2).std()
            )
            result['signal'] = price_direction * volume_std
            
            result['hft_trend'] = result.groupby('InstruID')['signal'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).apply(
                    lambda y: np.sum(y * np.exp(-np.arange(len(y))[::-1]/window))
                ))
            
            result.drop('signal', axis=1, inplace=True)
            
            return result

        @FactorManager.registry.register(
            name="microstructure_momentum",
            frequency=FactorFrequency.TICK,
            category="microstructure",
            description="微观结构动量因子"
        )
        def calculate_micro_momentum(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            
            result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
            
            result['mid_returns'] = result.groupby('InstruID')['mid_price'].transform(
                lambda x: x.pct_change()
            )
            result['volume_weight'] = result.groupby('InstruID')['Volume'].transform(
                lambda x: x / x.rolling(window=window, min_periods=window//2).sum()
            )
            result['microstructure_momentum'] = result['mid_returns'] * result['volume_weight']
            result['microstructure_momentum'] = result.groupby('InstruID')['microstructure_momentum'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).sum()
            )
            result.drop(['mid_price', 'mid_returns', 'volume_weight'], axis=1, inplace=True)
            
            return result

    @staticmethod
    def register_time_factors():
        """时间特征类因子"""

        @FactorManager.registry.register(
            name="intraday_seasonality",
            frequency=FactorFrequency.TICK,
            category="time",
            description="日内季节性因子"
        )
        def calculate_intraday_seasonality(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            result['minute'] = pd.to_datetime(result['UpdateTime']).dt.minute
            result['price_change'] = result.groupby('InstruID')['mid_price'].transform(
                lambda x: x.pct_change()
            )
            result['intraday_seasonality'] = result.groupby(['InstruID', 'minute'])['price_change'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            result.drop(['minute', 'price_change'], axis=1, inplace=True)
            return result

    @staticmethod
    def register_term_structure_factors():
        """期限结构类因子"""
        @FactorManager.registry.register(
            name="term_premium",
            frequency=FactorFrequency.TICK,
            category="term_structure",
            description="期限溢价因子"
        )
        def calculate_term_premium(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result['term_premium'] = result['mid_price'] * (252 / (result['days_to_expiry'] + 1))
            return result

    @staticmethod
    def register_composite_factors():
        """复合类因子"""
        @FactorManager.registry.register(
            name="volume_price_trend",
            frequency=FactorFrequency.TICK,
            category="composite",
            description="成交量加权价格趋势因子"
        )
        def calculate_volume_price_trend(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
            result = df.copy()
            result['price_trend'] = result.groupby('InstruID')['mid_price'].transform(
                lambda x: x.pct_change(window)
            )
            result['rel_volume'] = result.groupby('InstruID')['Volume'].transform(
                lambda x: x / x.rolling(window=window, min_periods=window//2).mean()
            )
            result['volume_price_trend'] = result['price_trend'] * result['rel_volume']
            result.drop(['price_trend', 'rel_volume'], axis=1, inplace=True)
            return result

        @FactorManager.registry.register(
            name="liquidity_adjusted_momentum",
            frequency=FactorFrequency.TICK,
            category="composite",
            description="流动性调整后的动量因子"
        )
        def calculate_liquidity_adjusted_momentum(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
            result = df.copy()
            result['momentum'] = result.groupby('InstruID')['mid_price'].transform(
                lambda x: x.pct_change(window)
            )
            result['liquidity_score'] = 1 / (result['spread'] * (1 + abs(result['depth_imbalance'])))
            result['liquidity_adjusted_momentum'] = result['momentum'] * result['liquidity_score']
            result.drop(['momentum', 'liquidity_score'], axis=1, inplace=True)
            return result


def register_all_factors():
    """注册所有因子"""
    register = FactorRegister()
    register.register_momentum_factors()      # 动量类因子
    register.register_volatility_factors()    # 波动率类因子
    register.register_volume_factors()        # 成交量类因子
    register.register_orderbook_factors()     # 订单簿类因子
    register.register_microstructure_factors()  # 市场微观结构类因子
    register.register_time_factors()          # 时间特征类因子
    register.register_term_structure_factors()  # 期限结构类因子
    register.register_composite_factors()      # 复合类因子