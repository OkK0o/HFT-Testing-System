import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
from scipy import stats
import warnings
import os
import time
from joblib import Parallel, delayed
from datetime import datetime
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class FactorTestConfig:
    """因子测试配置类"""
    ic_method: str = 'spearman'
    return_periods: List[int] = None
    n_quantiles: int = 20
    commission_rate: float = 0.0003
    min_sample_size: int = 30
    
    # IC筛选标准
    ic_threshold: float = 0.02  # IC均值最小阈值
    ir_threshold: float = 0.5   # IR最小阈值
    t_stat_threshold: float = 2.0  # t统计量最小阈值
    
    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = [1, 5, 10, 20]

class FactorsTester:
    """因子测试类"""
    def __init__(self, 
                 save_path: str,
                 config: Optional[FactorTestConfig] = None):
        """
        初始化因子测试器
        
        Args:
            save_path: 结果保存路径
            config: 测试配置，如果为None则使用默认配置
        """
        self.save_path = save_path
        self.config = config or FactorTestConfig()
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    def parallel_factor_test(self,
                           contracts: List[str],
                           factor_names: List[str],
                           n_jobs: int = 10,
                           method: str = 'spearman',
                           return_periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        并行计算多个合约的因子测试结果
        
        Args:
            contracts: 合约列表
            factor_names: 因子名称列表
            n_jobs: 并行进程数
            method: IC计算方法，'pearson'或'spearman'
            return_periods: 收益率周期列表
            
        Returns:
            因子测试结果DataFrame
        """
        start_time = time.time()
        
        # 任务数
        total_jobs = len(contracts)
        batch_size = max(1, (total_jobs + n_jobs - 1) // n_jobs)
        
        # 单个合约
        def process_single_contract(contract: str) -> pd.DataFrame:
            try:
                df = self._load_contract_data(contract)
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # 计算未来收益率
                df = self.calculate_forward_returns(df, return_periods)
                
                # 计算每个因子的IC
                results = []
                for factor in factor_names:
                    # 计算日度IC
                    daily_ics = []
                    for date in df['TradDay'].unique():
                        daily_data = df[df['TradDay'] == date]
                        ic_values = []
                        
                        for period in return_periods:
                            return_col = f'forward_return_{period}'
                            if factor not in daily_data.columns or return_col not in daily_data.columns:
                                ic_values.append(np.nan)
                                continue
                                
                            valid_data = daily_data[[factor, return_col]].dropna()
                            if len(valid_data) < 30:  # 最小样本量
                                ic_values.append(np.nan)
                                continue
                            # 对因子进行标准化处理
                            factor_std = (valid_data[factor] - valid_data[factor].mean()) / valid_data[factor].std()
                            
                            # 计算IC
                            if method == 'pearson':
                                ic = factor_std.corr(valid_data[return_col])
                            else:  # 'spearman'
                                ic = stats.spearmanr(factor_std, valid_data[return_col])[0]
                            
                            ic_values.append(ic)
                            
                        result_dict = {
                            'contract': contract,
                            'factor': factor,
                            'date': date
                        }
                        for period, ic in zip(return_periods, ic_values):
                            result_dict[f'ic_{period}'] = ic
                        
                        daily_ics.append(result_dict)
                    
                    results.extend(daily_ics)
                
                return pd.DataFrame(results)
            
            except Exception as e:
                warnings.warn(f"Error processing contract {contract}: {str(e)}")
                return pd.DataFrame()
        
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(process_single_contract)(contract) for contract in contracts
        )
        
        final_df = pd.concat(results, ignore_index=True)
        
        if not final_df.empty:
            save_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            for factor in factor_names:
                factor_df = final_df[final_df['factor'] == factor].copy()
                if not factor_df.empty:
                    filename = f"{factor}_{method}_ic_{save_time}.csv"
                    factor_df.to_csv(os.path.join(self.save_path, filename), index=False)
        
        end_time = time.time()
        print(f'因子测试完成，用时: {end_time - start_time:.2f}s')
        
        return final_df
    
    def _load_contract_data(self, contract: str) -> pd.DataFrame:
        """
        加载单个合约的数据
        
        Args:
            contract: 合约代码
            
        Returns:
            合约数据DataFrame
        """
        try:
            file_path = os.path.join(self.data_path, f"{contract}.feather")
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_feather(file_path)
            
            df['TradDay'] = pd.to_datetime(df['TradDay'].astype(str))
            mask = (df['TradDay'] >= pd.to_datetime(self.start_date)) & \
                   (df['TradDay'] <= pd.to_datetime(self.end_date))
            df = df[mask]
            
            return df
            
        except Exception as e:
            warnings.warn(f"Error loading contract {contract}: {str(e)}")
            return None
    
    @staticmethod
    def calculate_forward_returns(df: pd.DataFrame, 
                                periods: List[int],
                                price_col: str = 'mid_price',  # 修改默认使用中间价
                                time_col: str = None) -> pd.DataFrame:
        """
        计算未来收益率
        
        Args:
            df: 输入数据
            periods: 收益率周期列表
            price_col: 价格列名
            time_col: 时间列名，如果提供则按时间计算收益率，否则按行数计算
            
        Returns:
            添加了未来收益率的DataFrame
        """
        result = df.copy()
        
        for period in periods:
            if time_col is not None:
                result[f'{period}period_return'] = result.groupby('InstruID').apply(
                    lambda x: x[price_col].shift(-period) / x[price_col] - 1
                ).reset_index(level=0, drop=True)
            else:
                result[f'{period}period_return'] = result.groupby('InstruID')[price_col].transform(
                    lambda x: x.shift(-period) / x - 1
                )
        
        return result
    
    @staticmethod
    def calculate_ic(df: pd.DataFrame,
                    factor_names: Union[str, List[str]],
                    return_periods: List[int] = [1, 5, 10, 20],
                    method: str = 'pearson',
                    min_sample: int = 30) -> pd.DataFrame:
        """
        计算因子IC值
        
        Args:
            df: 输入数据
            factor_names: 因子名称或列表
            return_periods: 未来收益率周期列表
            method: 相关系数计算方法，'pearson'或'spearman'
            min_sample: 最小样本量
            
        Returns:
            IC值DataFrame，index为因子名称，columns为不同期限
        """
        if isinstance(factor_names, str):
            factor_names = [factor_names]
            
        ic_results = []
        
        for factor in factor_names:
            ic_values = []
            
            for period in return_periods:
                return_col = f'{period}period_return'
                
                if return_col not in df.columns:
                    warnings.warn(f"收益率列 {return_col} 不在数据中")
                    ic_values.append(np.nan)
                    continue
                
                # 按日期分组计算IC
                daily_ics = []
                for date in df['TradDay'].unique():
                    daily_data = df[df['TradDay'] == date].copy()
                    
                    # 去除缺失值
                    valid_data = daily_data[[factor, return_col]].dropna()
                    
                    if len(valid_data) < min_sample:
                        continue
                    
                    # 因子标准化和去极值
                    factor_values = valid_data[factor]
                    factor_std = (factor_values - factor_values.mean()) / factor_values.std()
                    
                    mask = np.abs(factor_std) <= 3
                    factor_std = factor_std[mask]
                    returns = valid_data[return_col][mask]
                    
                    if len(factor_std) < min_sample:
                        continue
                    
                    if method == 'pearson':
                        ic = factor_std.corr(returns)
                    else:  # 'spearman'
                        ic = stats.spearmanr(factor_std, returns)[0]
                    
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                
                if daily_ics:
                    ic_values.append(np.nanmean(daily_ics))
                else:
                    ic_values.append(np.nan)
            
            ic_results.append(ic_values)
        
        # 创建结果DataFrame
        ic_df = pd.DataFrame(ic_results,
                           index=factor_names,
                           columns=[f'{p}period_ic' for p in return_periods])
        
        return ic_df
    
    @staticmethod
    def calculate_ic_series(df: pd.DataFrame,
                           factor_names: Union[str, List[str]],
                           return_periods: List[int] = [1, 5, 10, 20],
                           method: str = 'pearson',
                           min_sample: int = 30) -> Dict[str, pd.DataFrame]:
        """
        计算因子IC值的时间序列
        
        Args:
            df: 输入数据
            factor_names: 因子名称或列表
            return_periods: 未来收益率周期列表
            method: 相关系数计算方法，'pearson'或'spearman'
            min_sample: 最小样本量
            
        Returns:
            字典，key为因子名称，value为该因子的IC时间序列DataFrame
        """
        if isinstance(factor_names, str):
            factor_names = [factor_names]
            
        ic_series_dict = {}
        
        for factor in factor_names:
            ic_series_list = []
            dates = df['TradDay'].unique()
            
            for date in dates:
                daily_data = df[df['TradDay'] == date]
                ic_values = []
                
                for period in return_periods:
                    return_col = f'{period}period_return'
                    
                    if return_col not in daily_data.columns:
                        ic_values.append(np.nan)
                        continue
                    
                    valid_data = daily_data[[factor, return_col]].dropna()
                    
                    if len(valid_data) < min_sample:
                        ic_values.append(np.nan)
                        continue
                    
                    if method == 'pearson':
                        ic = valid_data[factor].corr(valid_data[return_col])
                    else:  # 'spearman'
                        ic = stats.spearmanr(valid_data[factor], valid_data[return_col])[0]
                    
                    ic_values.append(ic)
                
                ic_series_list.append(ic_values)
            
            ic_df = pd.DataFrame(ic_series_list,
                               index=dates,
                               columns=[f'{p}period_ic' for p in return_periods])
            
            ic_series_dict[factor] = ic_df
        
        return ic_series_dict
    
    @staticmethod
    def calculate_ic_stats(ic_series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算IC序列的统计特征
        
        Args:
            ic_series_dict: 因子IC时间序列字典
            
        Returns:
            IC统计特征DataFrame
        """
        stats_list = []
        
        for factor_name, ic_df in ic_series_dict.items():
            factor_stats = []
            
            for column in ic_df.columns:
                ic_series = ic_df[column].dropna()
                
                stats = {
                    'mean': ic_series.mean(),
                    'std': ic_series.std(),
                    't_stat': (ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))),
                    'skew': ic_series.skew(),
                    'kurtosis': ic_series.kurtosis(),
                    'positive_ratio': (ic_series > 0).mean(),
                    'negative_ratio': (ic_series < 0).mean(),
                    'ir': ic_series.mean() / ic_series.std() if len(ic_series) > 0 else np.nan
                }
                
                factor_stats.append(pd.Series(stats))
            
            stats_df = pd.concat(factor_stats, axis=1)
            stats_df.columns = ic_df.columns
            stats_df.index = ['IC Mean', 'IC Std', 'T-Stat', 'Skewness', 
                            'Kurtosis', 'Positive Ratio', 'Negative Ratio', 'IR']
            
            stats_list.append(stats_df)
        
        result = pd.concat(stats_list, keys=ic_series_dict.keys(), axis=1)
        
        return result
    
    @staticmethod
    def plot_ic_series(ic_series_dict: Dict[str, pd.DataFrame],
                      factor_names: Union[str, List[str]] = None,
                      periods: List[int] = None) -> None:
        """
        绘制IC时间序列图
        
        Args:
            ic_series_dict: 因子IC时间序列字典
            factor_names: 要绘制的因子名称，如果为None则绘制所有因子
            periods: 要绘制的期限，如果为None则绘制所有期限
        """
        if factor_names is None:
            factor_names = list(ic_series_dict.keys())
        elif isinstance(factor_names, str):
            factor_names = [factor_names]
            
        plt.style.use('seaborn')
        
        for factor in factor_names:
            if factor not in ic_series_dict:
                warnings.warn(f"Factor {factor} not found in IC series dictionary")
                continue
                
            ic_df = ic_series_dict[factor]
            
            if periods is None:
                periods = [int(col.split('period')[0]) for col in ic_df.columns]
            
            fig, axes = plt.subplots(len(periods), 1, figsize=(15, 5*len(periods)))
            if len(periods) == 1:
                axes = [axes]
            
            for ax, period in zip(axes, periods):
                col = f'{period}period_ic'
                if col not in ic_df.columns:
                    warnings.warn(f"Period {period} not found for factor {factor}")
                    continue
                    
                ax.plot(ic_df.index, ic_df[col], label=f'{period}-Period IC')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                ax.fill_between(ic_df.index, 0, ic_df[col], 
                              alpha=0.3, where=ic_df[col]>0, color='g')
                ax.fill_between(ic_df.index, 0, ic_df[col], 
                              alpha=0.3, where=ic_df[col]<0, color='r')
                
                ax.set_title(f'{factor} {period}-Period IC Time Series')
                ax.set_xlabel('Date')
                ax.set_ylabel('IC Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # TODO: 多空开仓，Distrubution

    @staticmethod
    def plot_quantile_results(results: Dict[str, pd.DataFrame],
                            period: int,
                            factor_name: str) -> None:
        """
        绘制分位数回测结果
        
        Args:
            results: quantile_backtest返回的结果字典
            period: 收益率周期
            factor_name: 因子名称
        """
        if not all(key in results for key in ['quantile_returns', 'long_short_returns', 'performance_metrics']):
            print(f"警告：回测结果字典缺少必要的键")
            return
            
        if period not in results['quantile_returns']:
            print(f"警告：周期 {period} 的分位数收益率数据不存在")
            return
            
        if period not in results['long_short_returns']:
            print(f"警告：周期 {period} 的多空组合收益率数据不存在")
            return
            
        if period not in results['performance_metrics']:
            print(f"警告：周期 {period} 的绩效指标数据不存在")
            return
            
        quantile_returns = results['quantile_returns'][period]
        long_short_returns = results['long_short_returns'][period]
        performance = results['performance_metrics'][period]
        
        if quantile_returns.empty or long_short_returns.empty or performance.empty:
            print(f"警告：周期 {period} 的回测结果数据为空")
            return
            
        plt.style.use('seaborn')
        plt.rcParams['axes.unicode_minus'] = False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            ax1 = axes[0, 0]
            cum_returns = (1 + quantile_returns).cumprod()
            cum_returns.plot(ax=ax1)
            ax1.set_title(f'{factor_name} {period}-Period Cumulative Returns')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Returns')
            ax1.legend([f'Q{i+1}' for i in range(len(quantile_returns.columns))])
            ax1.grid(True)
            
            ax2 = axes[0, 1]
            cum_long_short = (1 + long_short_returns).cumprod()
            cum_long_short.plot(ax=ax2)
            ax2.set_title(f'{factor_name} {period}-Period Long-Short Portfolio Returns')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Returns')
            ax2.grid(True)
            
            ax3 = axes[1, 0]
            
            metric_names = {
                'cumulative_return': 'Cumulative Return',
                'annual_return': 'Annual Return',
                'annual_volatility': 'Annual Volatility',
                'sharpe_ratio': 'Sharpe Ratio',
                'max_drawdown': 'Max Drawdown',
                'win_rate': 'Win Rate'
            }
            performance.index = [metric_names.get(idx, idx) for idx in performance.index]
            
            # 重命名列名
            col_names = {f'quantile_{i}': f'Q{i+1}' for i in range(5)}
            col_names['long_short'] = 'L/S'
            performance.columns = [col_names.get(col, col) for col in performance.columns]
            
            sns.heatmap(performance, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax3)
            ax3.set_title(f'{factor_name} {period}-Period Performance Metrics')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.set_yticklabels(ax3.get_yticklabels())
            
            ax4 = axes[1, 1]
            mean_returns = quantile_returns.mean()
            mean_returns.plot(kind='bar', ax=ax4)
            ax4.set_title(f'{factor_name} {period}-Period Average Returns by Quantile')
            ax4.set_xlabel('Quantile')
            ax4.set_ylabel('Average Return')
            ax4.set_xticklabels([f'Q{i+1}' for i in range(len(mean_returns))], rotation=0)
            ax4.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"绘图时出错: {str(e)}")
            print("\n数据信息:")
            print(f"分位数收益率形状: {quantile_returns.shape}")
            print(f"多空收益率形状: {long_short_returns.shape}")
            print(f"绩效指标形状: {performance.shape}")

    @staticmethod
    def quantile_backtest(df: pd.DataFrame,
                         factor_name: str,
                         n_quantiles: int = 20,  # 改为20分位
                         return_periods: List[int] = [1, 5, 10, 20],
                         price_col: str = 'mid_price',  # 使用中间价
                         commission_rate: float = 0) -> Dict[str, pd.DataFrame]:
        """
        分位数因子开仓回测
        
        Args:
            df: 输入数据
            factor_name: 因子名称
            n_quantiles: 分位数数量，默认20分位
            return_periods: 收益率周期列表
            price_col: 价格列名，默认使用中间价
            commission_rate: 手续费率
            
        Returns:
            包含回测结果的字典，包括:
            - quantile_returns: 各分位数的收益率序列
            - long_short_returns: 多空组合的收益率序列（第一分位和第二分位的差值）
            - performance_metrics: 绩效指标
        """
        results = {}
        
        df = df.copy()
        df = df.dropna(subset=[factor_name])
        
        if df.empty:
            print(f"警告：{factor_name} 的所有数据都是无效的")
            return {
                'quantile_returns': {},
                'long_short_returns': {},
                'performance_metrics': {}
            }
            
        def safe_qcut(x):
            try:
                x = x.dropna()
                if len(x) < n_quantiles:
                    return pd.Series(index=x.index)
                    
                if x.nunique() == 1:
                    return pd.Series(0, index=x.index)
                    
                result = pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
                return result
                
            except Exception as e:
                print(f"警告：分位数计算出错 - {str(e)}")
                return pd.Series(index=x.index)
        
        df['quantile'] = df.groupby('TradDay')[factor_name].transform(safe_qcut)
        
        quantile_returns = {}
        long_short_returns = {}
        
        for period in return_periods:
            return_col = f'{period}period_return'
            if return_col not in df.columns:
                print(f"警告：{return_col} 不在数据列中")
                continue
            
            valid_data = df.dropna(subset=['quantile', return_col])
            if valid_data.empty:
                print(f"警告：{period}期数据全部无效")
                continue
            
            try:
                period_returns = valid_data.groupby(['TradDay', 'quantile'])[return_col].mean()
                if period_returns.empty:
                    print(f"警告：{period}期收益率计算结果为空")
                    continue
                period_returns = period_returns.unstack()
                if not period_returns.empty:
                    quantile_returns[period] = period_returns
                    # 修改多空组合构建方式：第一分位和第二分位的差值
                    first_quantile = period_returns[0]  # 第一分位
                    second_quantile = period_returns[1]  # 第二分位
                    
                    if first_quantile is not None and second_quantile is not None:
                        long_short = first_quantile - second_quantile
                        long_short = long_short - 2 * commission_rate
                        long_short_returns[period] = long_short
                    else:
                        print(f"警告：{period}期无法计算多空组合收益率")
                        
            except Exception as e:
                print(f"警告：计算{period}期收益率时出错 - {str(e)}")
                continue
        
        if not quantile_returns:
            print("警告：没有有效的分位数回测结果")
            return {
                'quantile_returns': {},
                'long_short_returns': {},
                'performance_metrics': {}
            }
        
        results['quantile_returns'] = quantile_returns
        results['long_short_returns'] = long_short_returns
        
        performance_metrics = {}
        for period in return_periods:
            if period not in quantile_returns or period not in long_short_returns:
                continue
                
            period_metrics = {}
            for q in quantile_returns[period].columns:
                returns = quantile_returns[period][q]
                if not returns.empty and not returns.isna().all():
                    metrics = FactorsTester._calculate_performance_metrics(returns)
                    period_metrics[f'quantile_{q}'] = metrics
            
            if not long_short_returns[period].empty and not long_short_returns[period].isna().all():
                long_short_metrics = FactorsTester._calculate_performance_metrics(
                    long_short_returns[period]
                )
                period_metrics['long_short'] = long_short_metrics
            
            if period_metrics:
                performance_metrics[period] = pd.DataFrame(period_metrics)
        
        results['performance_metrics'] = performance_metrics
        
        return results
    
    @staticmethod
    def _calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        计算收益率序列的绩效指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            包含各项绩效指标的字典
        """
        metrics = {}
        
        metrics['cumulative_return'] = (1 + returns).prod() - 1
        
        n_years = len(returns) / 252 
        metrics['annual_return'] = (1 + metrics['cumulative_return']) ** (1/n_years) - 1
        
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else np.nan
        
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()
        
        metrics['win_rate'] = (returns > 0).mean()
        
        return metrics

    def test_single_factor(self, 
                          factor_name: str, 
                          data: pd.DataFrame,
                          config: Optional[FactorTestConfig] = None) -> Dict:
        """
        单因子测试接口
        
        Args:
            factor_name: 因子名称
            data: 包含因子值的DataFrame
            config: 测试配置，如果为None则使用默认配置
            
        Returns:
            包含测试结果的字典
        """
        if config is None:
            config = self.config
            
        print(f"\n=== 开始测试因子: {factor_name} ===")
        
        ic_stats = self.calculate_ic(
            data,
            factor_names=factor_name,
            return_periods=config.return_periods,
            method=config.ic_method,
            min_sample=config.min_sample_size
        )
        
        ic_series = self.calculate_ic_series(
            data,
            factor_names=factor_name,
            return_periods=config.return_periods,
            method=config.ic_method,
            min_sample=config.min_sample_size
        )
        
        quantile_results = self.quantile_backtest(
            df=data,
            factor_name=factor_name,
            n_quantiles=config.n_quantiles,
            return_periods=config.return_periods,
            commission_rate=config.commission_rate
        )
        
        evaluation = self._evaluate_factor(ic_stats, ic_series[factor_name], config)
        
        results = {
            'factor_name': factor_name,
            'ic_stats': ic_stats,
            'ic_series': ic_series[factor_name],
            'quantile_results': quantile_results,
            'evaluation': evaluation
        }
        
        self._print_evaluation(results)
        
        max_period = max(config.return_periods)
        if max_period in quantile_results.get('quantile_returns', {}):
            self.plot_quantile_results(
                results=quantile_results,
                period=max_period,
                factor_name=factor_name
            )
        else:
            print(f"\n警告：无法绘制分位数回测结果，周期 {max_period} 的回测结果不完整")
        
        return results
    
    def _evaluate_factor(self, 
                        ic_stats: pd.DataFrame, 
                        ic_series: pd.DataFrame,
                        config: FactorTestConfig) -> Dict:
        """评估因子有效性"""
        evaluation = {
            'is_effective': False,
            'score': 0,
            'metrics': {},
            'comments': []
        }
        
        # 使用配置中指定的最大周期进行评估
        max_period = max(config.return_periods)
        ic_col = f'{max_period}period_ic'
        
        if ic_col not in ic_series.columns:
            print(f"警告：{ic_col} 不在IC序列中")
            return evaluation
        
        ic_values = ic_series[ic_col].dropna()
        if len(ic_values) == 0:
            print(f"警告：{ic_col} 的IC序列全为空值")
            return evaluation
        
        ic_mean = ic_values.mean()
        evaluation['metrics']['ic_mean'] = ic_mean
        if abs(ic_mean) >= config.ic_threshold:
            evaluation['score'] += 1
            evaluation['comments'].append(f"IC均值({ic_mean:.4f})通过阈值检验")
        else:
            evaluation['comments'].append(f"IC均值({ic_mean:.4f})未通过阈值检验")
            
        ic_std = ic_values.std()
        ir = ic_mean / ic_std if ic_std != 0 else 0
        evaluation['metrics']['ir'] = ir
        if abs(ir) >= config.ir_threshold:
            evaluation['score'] += 1
            evaluation['comments'].append(f"IR({ir:.4f})通过阈值检验")
        else:
            evaluation['comments'].append(f"IR({ir:.4f})未通过阈值检验")
            
        n_samples = len(ic_values)
        t_stat = ic_mean / (ic_std / np.sqrt(n_samples)) if ic_std != 0 else 0
        evaluation['metrics']['t_stat'] = t_stat
        if abs(t_stat) >= config.t_stat_threshold:
            evaluation['score'] += 1
            evaluation['comments'].append(f"t统计量({t_stat:.4f})通过阈值检验")
        else:
            evaluation['comments'].append(f"t统计量({t_stat:.4f})未通过阈值检验")
            
        ic_positive_ratio = (ic_values > 0).mean()
        evaluation['metrics']['ic_positive_ratio'] = ic_positive_ratio
        direction_stable = abs(ic_positive_ratio - 0.5) >= 0.1
        if direction_stable:
            evaluation['score'] += 1
            evaluation['comments'].append(f"IC方向稳定性良好(positive_ratio={ic_positive_ratio:.4f})")
        else:
            evaluation['comments'].append(f"IC方向不够稳定(positive_ratio={ic_positive_ratio:.4f})")
            
        evaluation['metrics']['n_samples'] = n_samples
        evaluation['comments'].append(f"有效样本量: {n_samples}")
            
        evaluation['is_effective'] = evaluation['score'] >= 3
        
        return evaluation
    
    def _print_evaluation(self, results: Dict):
        """打印因子评估结果"""
        print(f"\n=== {results['factor_name']} 因子评估结果 ===")
        eval_result = results['evaluation']
        
        print(f"\n有效性: {'有效' if eval_result['is_effective'] else '无效'}")
        print(f"评分: {eval_result['score']}/4")
        
        print("\n关键指标:")
        for metric, value in eval_result['metrics'].items():
            print(f"- {metric}: {value:.4f}")
            
        print("\n评估意见:")
        for comment in eval_result['comments']:
            print(f"- {comment}")




