import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict
import warnings
import datetime
from chinese_calendar import is_workday, is_holiday
import pytz

class FinDataProcessor:
    """金融数据处理器，专门用于处理期货高频数据"""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据文件所在目录
        """
        self.data_dir = Path(data_dir)
        self.cached_data = {}  # 用于缓存数据
        self._china_tz = pytz.timezone('Asia/Shanghai')
    
    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        加载单个feather文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的DataFrame
        """
        df = pd.read_feather(file_path)
        
        # 转换日期时间
        df['TradDay'] = pd.to_datetime(df['TradDay'].astype(str))
        df['DateTime'] = pd.to_datetime(
            df['TradDay'].dt.strftime('%Y-%m-%d') + ' ' + df['UpdateTime']
        )
        
        return df
    
    def load_data(self, 
                 start_date: Union[str, datetime.datetime],
                 end_date: Union[str, datetime.datetime],
                 contracts: Optional[List[str]] = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        加载指定日期范围的数据
        
        Args:
            start_date: 开始日期，格式：'YYYYMMDD'或datetime对象
            end_date: 结束日期，格式：'YYYYMMDD'或datetime对象
            contracts: 合约列表，如果为None则加载所有合约
            use_cache: 是否使用缓存
            
        Returns:
            合并后的DataFrame
        """
        # 转换日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 获取日期范围内的所有文件
        date_range = pd.date_range(start_date, end_date)
        file_paths = []
        for date in date_range:
            file_name = f"{date.strftime('%Y%m%d')}.feather"
            file_path = self.data_dir / file_name
            if file_path.exists():
                file_paths.append(file_path)
        
        # 加载数据
        dfs = []
        for file_path in file_paths:
            # 检查缓存
            cache_key = str(file_path)
            if use_cache and cache_key in self.cached_data:
                df = self.cached_data[cache_key]
            else:
                df = self._load_single_file(file_path)
                if use_cache:
                    self.cached_data[cache_key] = df
            
            # 如果指定了合约，则只保留指定合约的数据
            if contracts:
                df = df[df['InstruID'].isin(contracts)]
            
            dfs.append(df)
        
        # 合并数据
        if not dfs:
            raise ValueError(f"No data found between {start_date} and {end_date}")
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    
    def clean_data(self, 
                  df: pd.DataFrame,
                  columns: List[str] = None,
                  fill_method: str = 'ffill',
                  handle_outliers: bool = True) -> pd.DataFrame:
        """
        清洗数据
        
        如果指定了需要处理的列名,则只处理指定的列,否则处理所有数值型列
        Args:
            df: 输入DataFrame
            columns: 需要处理的列名
            fill_method: 填充方法，'ffill'/'bfill'/'interpolate'/'drop'
            handle_outliers: 是否处理异常值
            
        Returns:
            清洗后的DataFrame
        """
        df = df.copy()
        
        # 1. 处理重复数据
        df = df.drop_duplicates()
        
        # 2. 按时间和合约排序
        df = df.sort_values(['DateTime', 'InstruID'])
        
        # 3. 处理缺失值
        # 如果指定了列名就使用指定的列,否则使用所有数值型列
        if columns is not None:
            numeric_cols = [col for col in columns if col in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_method == 'drop':
            df = df.dropna(subset=numeric_cols)
        elif fill_method == 'ffill':
            df[numeric_cols] = df.groupby('InstruID')[numeric_cols].fillna(method='ffill')
        elif fill_method == 'bfill':
            df[numeric_cols] = df.groupby('InstruID')[numeric_cols].fillna(method='bfill')
        elif fill_method == 'interpolate':
            df[numeric_cols] = df.groupby('InstruID')[numeric_cols].apply(
                lambda x: x.interpolate(method='time')
            )
        
        # 4. 处理异常值（如果需要）
        if handle_outliers:
            # 如果指定了列名就使用指定的列,否则使用默认的价格列
            price_cols = ['LastPrice', 'HighPrice', 'LowPrice', 'OpenPrice']
            if columns is not None:
                cols_to_handle = [col for col in columns if col in df.columns]
            else:
                cols_to_handle = [col for col in price_cols if col in df.columns]
                
            for col in cols_to_handle:
                # 使用3个标准差作为阈值
                mean = df.groupby('InstruID')[col].transform('mean')
                std = df.groupby('InstruID')[col].transform('std')
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # 将异常值标记为NaN，然后用之前的方法填充
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df.loc[mask, col] = np.nan
                if fill_method == 'drop':
                    df = df.dropna(subset=[col])
                else:
                    df[col] = df.groupby('InstruID')[col].fillna(method='ffill')
        
        return df
    def add_features(self, 
                    df: pd.DataFrame,
                    features: List[str] = None) -> pd.DataFrame:
        """
        添加特征
        
        Args:
            df: 输入DataFrame
            features: 要添加的特征列表，可选：
                     - 'returns': 收益率
                     - 'vol': 波动率
                     - 'vwap': 成交量加权平均价格
                     - 'spread': 买卖价差
                     - 'depth_imbalance': 盘口深度不平衡
                     - 'mid_price': 中间价格
                     
        Returns:
            添加特征后的DataFrame
        """
        df = df.copy()
        
        if features is None:
            features = ['returns', 'vol', 'vwap', 'spread', 'depth_imbalance', 'mid_price']
        
        # 首先计算中间价，因为其他特征可能会用到
        if 'mid_price' in features:
            # 处理涨跌停情况下的中间价计算
            df['mid_price'] = np.where(
                (df['AskPrice1'] == 0) & (df['BidPrice1'] == 0),  # 两边都是0
                np.nan,  # 先设置为 NaN，后面用 ffill 填充
                np.where(
                    (df['AskPrice1'] == 0) & (df['BidPrice1'] != 0),  # 只有卖价为0
                    df['BidPrice1'],
                    np.where(
                        (df['BidPrice1'] == 0) & (df['AskPrice1'] != 0),  # 只有买价为0
                        df['AskPrice1'],
                        (df['AskPrice1'] + df['BidPrice1']) / 2  # 正常情况取均值
                    )
                )
            )
            # 按合约分组，用前值填充两边都是0的情况
            df['mid_price'] = df.groupby('InstruID')['mid_price'].fillna(method='ffill')
        
        for feature in features:
            if feature == 'returns':
                # 使用中间价计算收益率
                price_col = 'mid_price' if 'mid_price' in df.columns else 'LastPrice'
                df['returns'] = df.groupby('InstruID')[price_col].pct_change()
                
            elif feature == 'vol':
                # 使用中间价计算波动率
                window = 300  # 5分钟 = 300秒
                df['vol'] = df.groupby('InstruID')['returns'].rolling(
                    window=window, min_periods=1
                ).std().reset_index(0, drop=True)
                
            elif feature == 'vwap':
                # 计算VWAP
                df['turnover'] = df['LastPrice'] * df['Volume']
                df['vwap'] = (df.groupby('InstruID')['turnover'].cumsum() / 
                             df.groupby('InstruID')['Volume'].cumsum())
                
            elif feature == 'spread':
                # 计算买卖价差
                df['spread'] = df['AskPrice1'] - df['BidPrice1']
                
            elif feature == 'depth_imbalance':
                # 计算盘口深度不平衡
                bid_depth = df['BidVolume1'] + df['BidVolume2'] + df['BidVolume3']
                ask_depth = df['AskVolume1'] + df['AskVolume2'] + df['AskVolume3']
                df['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        
        return df
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加日历特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加日历特征后的DataFrame
        """
        df = df.copy()
        
        # 添加时间特征
        df['hour'] = df['DateTime'].dt.hour
        df['minute'] = df['DateTime'].dt.minute
        df['second'] = df['DateTime'].dt.second
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        
        # 标记交易时段
        df['session'] = 'other'
        morning_mask = ((df['hour'] == 9) & (df['minute'] >= 30)) | \
                      ((df['hour'] == 10)) | \
                      ((df['hour'] == 11) & (df['minute'] <= 30))
        afternoon_mask = ((df['hour'] >= 13) & (df['hour'] < 15)) | \
                        ((df['hour'] == 15) & (df['minute'] <= 15))
        df.loc[morning_mask, 'session'] = 'morning'
        df.loc[afternoon_mask, 'session'] = 'afternoon'
        
        # 标记节假日
        df['is_holiday'] = df['TradDay'].apply(
            lambda x: is_holiday(x.date())
        )
        
        # 标记是否跨日
        df['is_overnight'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 6, 7])
        
        # 计算到期天数（针对期货合约）
        def get_expiry_date(instru_id: str) -> datetime.datetime:
            """从合约ID获取到期日期"""
            if not instru_id.startswith(('IF', 'IH', 'IC')):
                return None
            
            year = int('20' + instru_id[-4:-2])
            month = int(instru_id[-2:])
            
            # 获取该月第三个周五
            first_day = datetime.datetime(year, month, 1)
            weekday = first_day.weekday()
            friday_count = 0
            day = 1
            while friday_count < 3:
                if (weekday + day - 1) % 7 == 4:  # 4 represents Friday
                    friday_count += 1
                if friday_count < 3:
                    day += 1
            
            return datetime.datetime(year, month, day)
        
        df['expiry_date'] = df['InstruID'].apply(get_expiry_date)
        df['days_to_expiry'] = (df['expiry_date'] - df['DateTime']).dt.days
        
        return df
    
    def resample_data(self,
                     df: pd.DataFrame,
                     freq: str = '1min',
                     price_col: str = 'mid_price',  # 默认使用中间价
                     volume_col: str = 'Volume',
                     agg_dict: Dict = None) -> pd.DataFrame:
        """
        重采样数据
        
        Args:
            df: 输入DataFrame
            freq: 重采样频率，如'1min', '5min', '1H'
            price_col: 价格列名，默认使用中间价
            volume_col: 成交量列名
            agg_dict: 自定义聚合字典
            
        Returns:
            重采样后的DataFrame
        """
        df = df.copy()
        
        # 如果没有中间价，先计算中间价
        if price_col == 'mid_price' and 'mid_price' not in df.columns:
            df = self.add_features(df, features=['mid_price'])
        
        # 默认的聚合方式
        if agg_dict is None:
            agg_dict = {
                price_col: 'last',                    # 最新价取最后一个
                'HighPrice': 'max',                   # 最高价取最大值
                'LowPrice': 'min',                    # 最低价取最小值
                'OpenPrice': 'first',                 # 开盘价取第一个
                volume_col: 'sum',                    # 成交量求和
                'Turnover': 'sum',                    # 成交额求和
                'BidPrice1': 'last',                 # 买一价取最后一个
                'AskPrice1': 'last',                 # 卖一价取最后一个
                'BidVolume1': 'last',               # 买一量取最后一个
                'AskVolume1': 'last',               # 卖一量取最后一个
                'mid_price': 'last'                 # 中间价取最后一个
            }
        
        # 按合约分组重采样
        resampled_dfs = []
        for instru_id, group in df.groupby('InstruID'):
            # 设置时间索引
            group = group.set_index('DateTime')
            
            # 重采样
            resampled = group.resample(freq).agg(agg_dict)
            resampled['InstruID'] = instru_id
            
            resampled_dfs.append(resampled)
        
        # 合并结果
        result = pd.concat(resampled_dfs, axis=0)
        result = result.reset_index()
        
        return result

    def clear_cache(self):
        """清除缓存数据"""
        self.cached_data.clear() 
   
    # TODO：多合约聚合