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

    @staticmethod
    def register_minute_factors():
        """分钟级别因子 - 使用至少120期历史数据"""
        minute_windows = [120, 240, 360, 480]  # 2小时、4小时、6小时、8小时对应的数据条数
        
        # 1. 分钟级别多周期动量因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_momentum_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟动量因子"
            )
            def calculate_minute_momentum(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                # 确保有mid_price列
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                result[f'minute_momentum_{w}'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(w)
                )
                return result
        
        # 2. 分钟级别波动率持续性因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_vol_persistence_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volatility",
                description=f"{window}分钟波动率持续性因子"
            )
            def calculate_minute_vol_persistence(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算收益率
                result['returns'] = result.groupby('InstruID')['mid_price'].transform(lambda x: x.pct_change())
                
                # 计算短期波动率和长期波动率的比值
                short_window = max(w // 4, 30)  # 短期窗口为长期窗口的1/4，最小30
                
                result['short_vol'] = result.groupby('InstruID')['returns'].transform(
                    lambda x: x.rolling(window=short_window, min_periods=short_window//2).std()
                )
                result['long_vol'] = result.groupby('InstruID')['returns'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).std()
                )
                
                result[f'minute_vol_persistence_{w}'] = result['short_vol'] / (result['long_vol'] + 1e-8)
                
                # 删除临时列
                result = result.drop(['returns', 'short_vol', 'long_vol'], axis=1)
                return result
        
        # 3. 分钟级别价格反转因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_reversal_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟价格反转因子"
            )
            def calculate_minute_reversal(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算短期和长期收益率
                short_window = max(w // 6, 20)  # 短期窗口，最小20
                
                result['short_return'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(short_window)
                )
                result['long_return'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(w)
                )
                
                # 反转因子 = 短期收益率 - 长期收益率的一部分
                result[f'minute_reversal_{w}'] = result['short_return'] - 0.5 * result['long_return']
                
                # 删除临时列
                result = result.drop(['short_return', 'long_return'], axis=1)
                return result
        
        # 4. 分钟级别成交量价格相关性因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_volume_price_corr_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volume",
                description=f"{window}分钟成交量价格相关性因子"
            )
            def calculate_minute_volume_price_corr(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                result['price_change'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change()
                )
                
                # 计算滚动相关性
                def rolling_correlation(group):
                    price_changes = group['price_change']
                    volumes = group['Volume']
                    corrs = pd.Series(index=group.index)
                    
                    for i in range(len(group)):
                        if i >= w:
                            # 使用前w个观测值计算相关性
                            corr = price_changes.iloc[i-w:i].corr(volumes.iloc[i-w:i])
                            corrs.iloc[i] = corr
                    
                    return corrs
                
                # 分组计算相关性
                result[f'minute_volume_price_corr_{w}'] = result.groupby('InstruID').apply(
                    rolling_correlation
                ).reset_index(level=0, drop=True)
                
                # 删除临时列
                result = result.drop(['price_change'], axis=1)
                return result
        
        # 5. 分钟级别价格动量加速度因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_momentum_acceleration_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟动量加速度因子"
            )
            def calculate_minute_momentum_acceleration(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算连续两个时间段的动量变化
                medium_window = w // 2  # 中期窗口
                
                result['recent_momentum'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(medium_window)
                )
                result['older_momentum'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.shift(medium_window).pct_change(medium_window)
                )
                
                # 动量加速度 = 近期动量 - 早期动量
                result[f'minute_momentum_acceleration_{w}'] = result['recent_momentum'] - result['older_momentum']
                
                # 删除临时列
                result = result.drop(['recent_momentum', 'older_momentum'], axis=1)
                return result
        
        # 6. 分钟级别价格趋势强度因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_trend_strength_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟趋势强度因子"
            )
            def calculate_minute_trend_strength(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算线性回归斜率
                def calculate_trend_slope(x):
                    if len(x) < w//2:
                        return np.nan
                    
                    y = x.values
                    t = np.arange(len(y))
                    
                    # 计算最小二乘线性回归的斜率
                    slope, _, r_value, _, _ = stats.linregress(t, y)
                    
                    # 趋势强度 = 斜率 * R²
                    trend_strength = slope * (r_value ** 2)
                    return trend_strength
                
                # 计算每个标的的趋势强度
                result[f'minute_trend_strength_{w}'] = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).apply(calculate_trend_slope, raw=False)
                )
                
                return result

        # 7. 分钟级别RSI因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_rsi_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟相对强弱指标"
            )
            def calculate_minute_rsi(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算价格变化
                delta = result.groupby('InstruID')['mid_price'].transform(lambda x: x.diff())
                
                # 计算上涨和下跌
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # 计算平均上涨和下跌
                avg_gain = gain.rolling(window=w, min_periods=w//2).mean()
                avg_loss = loss.rolling(window=w, min_periods=w//2).mean()
                
                # 计算RSI
                rs = avg_gain / (avg_loss + 1e-9)
                result[f'minute_rsi_{w}'] = 100 - (100 / (1 + rs))
                
                return result
        
        # 8. 分钟级别MACD因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_macd_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟MACD指标"
            )
            def calculate_minute_macd(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算EMA
                ema12 = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.ewm(span=w//2, min_periods=w//4).mean()
                )
                ema26 = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.ewm(span=w, min_periods=w//2).mean()
                )
                
                # 计算MACD线和信号线
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=w//3, min_periods=w//6).mean()
                
                result[f'minute_macd_{w}'] = macd_line - signal_line
                
                return result
        
        # 9. 分钟级别布林带因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_bollinger_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volatility",
                description=f"{window}分钟布林带指标"
            )
            def calculate_minute_bollinger(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算中轨和标准差
                middle_band = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).mean()
                )
                std = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).std()
                )
                
                # 计算布林带宽度
                result[f'minute_bollinger_{w}'] = (result['mid_price'] - middle_band) / (std + 1e-9)
                
                return result
        
        # 10. 分钟级别成交量加权价格趋势因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_vwap_trend_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volume",
                description=f"{window}分钟成交量加权价格趋势"
            )
            def calculate_minute_vwap_trend(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算VWAP
                vwap = result.groupby('InstruID').apply(
                    lambda x: (x['mid_price'] * x['Volume']).rolling(window=w, min_periods=w//2).sum() / 
                            x['Volume'].rolling(window=w, min_periods=w//2).sum()
                ).reset_index(level=0, drop=True)
                
                # 计算VWAP趋势
                result[f'minute_vwap_trend_{w}'] = (result['mid_price'] - vwap) / (vwap + 1e-9)
                
                return result
        
        # 11. 分钟级别价格动量背离因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_momentum_divergence_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟价格动量背离指标"
            )
            def calculate_minute_momentum_divergence(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算价格动量和动量变化
                momentum = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change(w//2)
                )
                momentum_change = momentum.diff()
                
                # 计算价格变化
                price_change = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change()
                )
                
                # 计算背离
                result[f'minute_momentum_divergence_{w}'] = np.sign(momentum_change) * np.sign(price_change)
                
                return result
        
        # 12. 分钟级别价格波动率比率因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_volatility_ratio_{window}",
                frequency=FactorFrequency.MINUTE,
                category="volatility",
                description=f"{window}分钟价格波动率比率"
            )
            def calculate_minute_volatility_ratio(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算短期和长期波动率
                short_vol = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.rolling(window=w//4, min_periods=w//8).std()
                )
                long_vol = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.rolling(window=w, min_periods=w//2).std()
                )
                
                # 计算波动率比率
                result[f'minute_volatility_ratio_{w}'] = short_vol / (long_vol + 1e-9)
                
                return result
        
        # 13. 分钟级别价格趋势加速度因子
        for window in minute_windows:
            @FactorManager.registry.register(
                name=f"minute_trend_acceleration_{window}",
                frequency=FactorFrequency.MINUTE,
                category="momentum",
                description=f"{window}分钟价格趋势加速度"
            )
            def calculate_minute_trend_acceleration(df: pd.DataFrame, w=window) -> pd.DataFrame:
                result = df.copy()
                if 'mid_price' not in result.columns:
                    result['mid_price'] = (result['AskPrice1'] + result['BidPrice1']) / 2
                
                # 计算价格变化率
                price_change = result.groupby('InstruID')['mid_price'].transform(
                    lambda x: x.pct_change()
                )
                
                # 计算加速度
                result[f'minute_trend_acceleration_{w}'] = price_change.rolling(
                    window=w//2, min_periods=w//4
                ).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
                
                return result

def register_all_factors(register_smoothed=False, smoothing_window=1200):
    """注册所有因子"""
    FactorRegister.register_momentum_factors()
    FactorRegister.register_volatility_factors()
    FactorRegister.register_volume_factors()
    FactorRegister.register_orderbook_factors()
    FactorRegister.register_microstructure_factors()
    FactorRegister.register_time_factors()
    FactorRegister.register_term_structure_factors()
    FactorRegister.register_composite_factors()
    FactorRegister.register_minute_factors()  # 添加分钟级别因子注册
    
    # 注册平滑因子
    if register_smoothed:
        factor_names = FactorManager.get_factor_names()
        register_smoothed_factors(smoothing_window, 'ema', factor_names)

def register_smoothed_factors(window=1200, method='ema', factor_names=None):
    """
    为指定的因子创建平滑版本
    
    Args:
        window: 平滑窗口大小，默认1200
        method: 平滑方法，'ema'或'sma'，默认'ema'
        factor_names: 要平滑的因子名称列表，如果为None则平滑所有因子
        
    Returns:
        平滑因子名称列表
    """
    # 确保先注册所有基础因子
    register_all_factors()
    
    # 获取要平滑的因子列表
    if factor_names is None:
        # 如果未指定因子列表，则获取所有已注册的因子
        all_factors = FactorManager.get_factor_names()
        # 过滤掉已经是平滑因子的因子
        original_factors = [f for f in all_factors if '_ema' not in f and '_sma' not in f]
    else:
        # 使用指定的因子列表
        original_factors = [f for f in factor_names if '_ema' not in f and '_sma' not in f]
    
    print(f"为{len(original_factors)}个因子创建{method.upper()}{window}平滑版本:")
    
    # 为每个因子注册平滑版本
    smoothed_factors = []
    for factor in original_factors:
        try:
            # 检查因子是否已注册
            if factor not in FactorManager.get_factor_names():
                print(f"  - 跳过因子 {factor}: 未注册")
                continue
                
            smoothed_name = FactorManager.register_smoothed_factor(
                original_name=factor,
                window=window,
                method=method
            )
            smoothed_factors.append(smoothed_name)
            print(f"  - 创建平滑因子: {smoothed_name}")
        except Exception as e:
            print(f"  - 无法为因子 {factor} 创建平滑版本: {str(e)}")
    
    print(f"成功创建{len(smoothed_factors)}个平滑因子")
    return smoothed_factors