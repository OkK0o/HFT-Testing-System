from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from backtesting.lib import crossover
from backtesting.test import SMA
from dataclasses import dataclass
import json
import os
import warnings
from backtesting.lib import OHLCV_AGG, resample_apply

# 设置固定随机种子以便结果可重现
np.random.seed(42)

class SignalStrategy(Strategy):
    """
    基于信号的交易策略
    
    信号约定:
    - 1: 买入信号
    - 0: 卖出信号
    - -1: 不交易信号
    """
    
    # 风控参数
    use_stoploss = True       # 是否使用止损
    sl_percent = 0.008        # 止损百分比
    use_takeprofit = True     # 是否使用止盈
    tp_percent = 0.02         # 止盈百分比
    position_size = 0.05      # 每次用资金比例
    max_daily_loss = 0.01     # 日内最大亏损百分比
    
    # 信号确认参数
    confirm_threshold = 1     # 需要连续多少个相同信号才确认，改回1
    cool_down_hours = 4       # 止损后冷却时间，从12小时减少到4小时
    
    # 交易逻辑参数
    max_long_positions = 2    # 最大多头持仓数量，从1增加到2
    max_short_positions = 2   # 最大空头持仓数量，从1增加到2
    
    # 新增参数
    require_reverse_signal = False  # 是否要求反向信号才能开仓
    
    def init(self):
        """初始化方法"""
        # 设置交易回调
        self.trade_exit_callbacks = []
        
        # 日内PnL计算
        self.daily_pnl = 0
        self.current_day = None
        
        # 风控状态
        self.cool_down_until = None
        
        # 持仓统计
        self.long_positions = 0
        self.short_positions = 0
        
        # 交易记录
        self.last_trade_time = None
        self.last_trade_side = None  # 1为多头, 0为空头
        
        # 信号确认计数
        self.long_confirm_count = 0
        self.short_confirm_count = 0
        
        # 反向信号逻辑
        self.waiting_for_reverse_signal = False
        self.last_closed_side = None  # 记录最近平仓的交易方向
        
        # 注册交易回调
        self.add_trade_exit_callback(self.on_trade_close)
        
        # 确保交易回调被引擎正确处理
        try:
            if hasattr(self, '_broker'):
                self._broker._trade_callbacks.append(self._notify_trade)
                self._broker._trade_on_close = True
        except Exception as e:
            print(f"警告: 无法设置交易回调: {e}")
    
    def add_trade_exit_callback(self, callback):
        """添加交易结束回调函数"""
        self.trade_exit_callbacks.append(callback)
    
    def _notify_trade(self, trade):
        """
        通知交易状态变化的内部方法
        当交易平仓时调用所有注册的回调函数
        """
        # 只处理已平仓的交易
        if trade.is_closed:
            for callback in self.trade_exit_callbacks:
                callback(trade)
    
    def on_trade_close(self, trade):
        """
        交易关闭回调函数
        
        Args:
            trade: 已关闭的交易对象
        """
        current_time = self.data.index[-1]
        profit_pct = trade.pl_pct * 100
        
        # 更新日内P&L
        current_day = current_time.date()
        if self.current_day != current_day:
            self.current_day = current_day
            self.daily_pnl = 0
        
        self.daily_pnl += trade.pl
        
        # 输出交易结果
        print(f"时间: {current_time}, 交易结束, 方向: {'多头' if trade.size > 0 else '空头'}, "
              f"收益: {trade.pl:.2f} ({profit_pct:.2f}%), 持续时间: {trade.entry_time} ~ {trade.exit_time}")
        
        # 更新持仓计数
        if trade.size > 0:
            self.long_positions -= 1
            # 记录关闭的交易方向
            self.last_closed_side = 1
            # 只在需要反向信号时设置等待标志
            if self.require_reverse_signal:
                self.waiting_for_reverse_signal = True
        else:
            self.short_positions -= 1
            # 记录关闭的交易方向
            self.last_closed_side = 0
            # 只在需要反向信号时设置等待标志
            if self.require_reverse_signal:
                self.waiting_for_reverse_signal = True
        
        # 风控逻辑 - 如果日内亏损超过设定值，触发冷却时间
        if self.daily_pnl < -self.max_daily_loss * self.equity:
            self.cool_down_until = current_time + pd.Timedelta(hours=self.cool_down_hours)
            print(f"警告: 日内亏损达到上限，暂停交易至 {self.cool_down_until}")
    
    def next(self):
        """每根K线执行一次的主方法"""
        current_time = self.data.index[-1]
        current_signal = self.data.Signal[-1]
        
        # 添加在next方法的开头部分
        if current_time.day % 5 == 0 and current_time.hour == 10:  # 每5天记录一次
            print(f"当前时间: {current_time}, 信号: {current_signal}, 多头持仓: {self.long_positions}, 空头持仓: {self.short_positions}")
        
        # 风控检查 - 是否在冷却期
        if self.cool_down_until and current_time < self.cool_down_until:
            return
        
        # 如果没有持仓且信号为-1(不交易信号)，则跳过
        if not self.position and current_signal == -1:
            return
        
        # 信号确认逻辑
        if current_signal == 1:  # 多头信号
            self.long_confirm_count += 1
            self.short_confirm_count = 0
        elif current_signal == 0:  # 空头信号
            self.short_confirm_count += 1
            self.long_confirm_count = 0
        else:
            self.long_confirm_count = 0
            self.short_confirm_count = 0
        
        # 当已有持仓，看是否需要平仓
        if self.position:
            # 简单的平仓逻辑 - 收到反向信号时平仓
            if self.position.is_long and current_signal == 0 and self.short_confirm_count >= self.confirm_threshold:
                self.position.close()
                print(f"时间: {current_time}, 平多头仓位, 价格: {self.data.Close[-1]:.2f}")
            elif self.position.is_short and current_signal == 1 and self.long_confirm_count >= self.confirm_threshold:
                self.position.close()
                print(f"时间: {current_time}, 平空头仓位, 价格: {self.data.Close[-1]:.2f}")
            return
            
        # 开仓逻辑 - 根据信号和确认阈值
        
        # 检查是否需要反向信号
        if self.waiting_for_reverse_signal:
            # 如果上一笔是多头交易，则需要空头信号才能开仓
            if self.last_closed_side == 1 and current_signal == 0 and self.short_confirm_count >= self.confirm_threshold:
                self.waiting_for_reverse_signal = False
                # 重置标志，已收到反向信号
                print(f"时间: {current_time}, 收到反向信号(空头)，已重置交易锁定")
            # 如果上一笔是空头交易，则需要多头信号才能开仓
            elif self.last_closed_side == 0 and current_signal == 1 and self.long_confirm_count >= self.confirm_threshold:
                self.waiting_for_reverse_signal = False
                # 重置标志，已收到反向信号
                print(f"时间: {current_time}, 收到反向信号(多头)，已重置交易锁定")
            
            # 如果仍在等待反向信号，则跳过开仓操作
            if self.waiting_for_reverse_signal:
                return
        
        # 检查距离上次交易的时间间隔，至少需要间隔1根K线
        if self.last_trade_time is not None:
            time_diff = current_time - self.last_trade_time
            if time_diff.total_seconds() < 300:  # 5分钟K线，确保至少间隔1根
                return
        
        if current_signal == 1 and self.long_confirm_count >= self.confirm_threshold:  # 多头信号
            # 检查多头持仓数量是否已达到上限
            if self.long_positions >= self.max_long_positions:
                return
            
            # 计算止损和止盈价格
            entry_price = self.data.Close[-1]
            sl_price = entry_price * (1 - self.sl_percent) if self.use_stoploss else None
            tp_price = entry_price * (1 + self.tp_percent) if self.use_takeprofit else None
            
            # 格式化止损止盈显示字符串
            sl_price_str = "无" if sl_price is None else f"{sl_price:.2f}"
            tp_price_str = "无" if tp_price is None else f"{tp_price:.2f}"
            
            # 开多头仓位
            self.buy(size=self.position_size, sl=sl_price, tp=tp_price)
            print(f"时间: {current_time}, 多头开仓, 价格: {entry_price:.2f}, 止损: {sl_price_str}, 止盈: {tp_price_str}")
            self.last_trade_side = 1
            self.long_positions += 1
            self.last_trade_time = current_time
            
        elif current_signal == 0 and self.short_confirm_count >= self.confirm_threshold:  # 空头信号
            # 检查空头持仓数量是否已达到上限
            if self.short_positions >= self.max_short_positions:
                return
                
            # 计算止损和止盈价格
            entry_price = self.data.Close[-1]
            sl_price = entry_price * (1 + self.sl_percent) if self.use_stoploss else None
            tp_price = entry_price * (1 - self.tp_percent) if self.use_takeprofit else None
            
            # 格式化止损止盈显示字符串
            sl_price_str = "无" if sl_price is None else f"{sl_price:.2f}"
            tp_price_str = "无" if tp_price is None else f"{tp_price:.2f}"
            
            # 开空头仓位
            self.sell(size=self.position_size, sl=sl_price, tp=tp_price)
            print(f"时间: {current_time}, 空头开仓, 价格: {entry_price:.2f}, 止损: {sl_price_str}, 止盈: {tp_price_str}")
            self.last_trade_side = 0
            self.short_positions += 1
            self.last_trade_time = current_time


def prepare_data(csv_file, signal_file):
    """
    准备回测数据
    
    Args:
        csv_file: 分钟级数据文件路径
        signal_file: 信号文件路径
    
    Returns:
        准备好的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"分钟数据文件不存在: {csv_file}")
    if not os.path.exists(signal_file):
        raise FileNotFoundError(f"信号文件不存在: {signal_file}")
    
    print(f"开始读取数据文件...")
    
    try:
        # 读取分钟级数据
        data = pd.read_csv(csv_file)
        # 读取信号数据
        signals = pd.read_csv(signal_file)
        
        # 检查必要的列是否存在
        required_data_cols = ['TradDay', 'UpdateTime']
        required_signal_cols = ['TradDay', 'UpdateTime', 'Pred_Label']
        
        for col in required_data_cols:
            if col not in data.columns:
                raise ValueError(f"分钟数据缺少必要的列: {col}")
        
        for col in required_signal_cols:
            if col not in signals.columns:
                raise ValueError(f"信号数据缺少必要的列: {col}")
        
        # 检查数据时间范围
        print(f"\n数据时间范围分析:")
        print(f"分钟数据日期范围: {data['TradDay'].min()} 到 {data['TradDay'].max()}")
        print(f"信号数据日期范围: {signals['TradDay'].min()} 到 {signals['TradDay'].max()}")
        
        # 检查数据量是否足够
        if len(data) < 100:
            raise ValueError(f"分钟数据数量过少: {len(data)}行")
        if len(signals) < 10:
            raise ValueError(f"信号数据数量过少: {len(signals)}行")
        
        # 绘制信号分布直方图
        plt.figure(figsize=(10, 6))
        signals['Pred_Label'].hist(bins=3)
        plt.title('信号分布直方图')
        plt.xlabel('信号值')
        plt.ylabel('频率')
        plt.savefig('signal_histogram.png')
        print("信号分布直方图已保存为signal_histogram.png")
        
        # 按日期绘制信号变化
        signals_by_day = signals.groupby('TradDay')['Pred_Label'].agg(['mean', 'nunique'])
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(signals_by_day.index, signals_by_day['mean'])
        plt.title('每日平均信号值')
        plt.ylabel('平均信号')
        
        plt.subplot(2, 1, 2)
        plt.bar(signals_by_day.index, signals_by_day['nunique'])
        plt.title('每日不同信号值数量')
        plt.ylabel('不同信号数量')
        plt.tight_layout()
        plt.savefig('signal_by_day.png')
        print("每日信号变化图已保存为signal_by_day.png")
        
        # 信号值分析
        print(f"\n信号值分析:")
        print(f"信号文件中的Pred_Label分布:\n{signals['Pred_Label'].value_counts()}")
        print(f"信号文件中的不同日期数量: {signals['TradDay'].nunique()}")
        
        # 检查信号是否有变化
        signal_changes = signals['Pred_Label'].diff().ne(0).sum()
        print(f"信号文件中的信号变化次数: {signal_changes}")
        
        # 处理日期和时间，添加异常处理
        try:
            data['TradDay'] = pd.to_datetime(data['TradDay'])
            signals['TradDay'] = pd.to_datetime(signals['TradDay'])
        except Exception as e:
            print(f"警告: 日期转换错误: {e}")
            # 尝试不同的日期格式
            try:
                data['TradDay'] = pd.to_datetime(data['TradDay'], format='%Y%m%d')
                signals['TradDay'] = pd.to_datetime(signals['TradDay'], format='%Y%m%d')
                print("使用替代日期格式成功转换")
            except Exception as e2:
                raise ValueError(f"无法转换日期格式: {e2}")
        
        # 处理UpdateTime确保格式一致
        if 'UpdateTime' in data.columns:
            try:
                data['UpdateTime'] = pd.to_datetime(data['UpdateTime']).dt.time
            except:
                # 尝试常见时间格式
                try:
                    data['UpdateTime'] = pd.to_datetime(data['UpdateTime'], format='%H:%M:%S').dt.time
                except:
                    print("警告: 无法解析UpdateTime，尝试其他格式")
        
        if 'UpdateTime' in signals.columns:
            try:
                signals['UpdateTime'] = pd.to_datetime(signals['UpdateTime']).dt.time
            except:
                try:
                    signals['UpdateTime'] = pd.to_datetime(signals['UpdateTime'], format='%H:%M:%S').dt.time
                except:
                    print("警告: 无法解析信号的UpdateTime，尝试其他格式")
        
        # 创建唯一标识符用于合并
        data['date_time_key'] = data['TradDay'].dt.strftime('%Y-%m-%d') + '_' + data['UpdateTime'].astype(str)
        signals['date_time_key'] = signals['TradDay'].dt.strftime('%Y-%m-%d') + '_' + signals['UpdateTime'].astype(str)
        
        # 合并数据与信号
        print("合并数据和信号...")
        merged_data = pd.merge(data, signals[['date_time_key', 'Pred_Label']], 
                              on='date_time_key', how='inner')  # 使用inner join确保只使用有信号的数据
        
        print(f"合并后的数据点数: {len(merged_data)}")
        
        # 检查合并后的数据量
        if len(merged_data) == 0:
            raise ValueError("合并后的数据为空，请检查日期和时间格式是否匹配")
        
        # 创建时间索引
        try:
            merged_data['Timestamp'] = pd.to_datetime(
                merged_data['TradDay'].astype(str) + ' ' + 
                merged_data['UpdateTime'].astype(str)
            )
            merged_data.set_index('Timestamp', inplace=True)
        except Exception as e:
            print(f"创建时间索引失败: {e}")
            # 尝试替代方法
            merged_data['Timestamp'] = merged_data['TradDay']
            merged_data.set_index('Timestamp', inplace=True)
            print("使用简化的时间索引")
        
        # 确保数据有必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in merged_data.columns:
                # 如果没有对应的列，尝试映射常见的列名
                mapping = {
                    'Open': ['open', 'Open', 'open_price', 'OpenPrice'],
                    'High': ['high', 'High', 'high_price', 'HighPrice'],
                    'Low': ['low', 'Low', 'low_price', 'LowPrice'],
                    'Close': ['close', 'Close', 'close_price', 'ClosePrice', 'mid_price', 'LastPrice'],
                    'Volume': ['volume', 'Volume', 'vol', 'Vol']
                }
                
                # 查找匹配的列名
                found = False
                for alt_name in mapping[col]:
                    if alt_name in merged_data.columns:
                        merged_data[col] = merged_data[alt_name]
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"数据中缺少必要的列: {col}")
        
        # 重命名信号列并确保为整数类型
        merged_data.rename(columns={'Pred_Label': 'Signal'}, inplace=True)
        merged_data['Signal'] = merged_data['Signal'].astype(int)
        
        # 只保留必要的列
        columns_to_keep = required_columns + ['Signal']
        merged_data = merged_data[columns_to_keep]
        
        # 检查时间顺序和连续性
        merged_data = merged_data.sort_index()
        
        # 检查并处理NaN值
        nan_count_before = merged_data.isna().sum().sum()
        if nan_count_before > 0:
            print(f"数据中有NaN值: {nan_count_before}个")
            print("各列NaN值统计:")
            print(merged_data.isna().sum())
            
            # 填充缺失值
            for col in required_columns:
                # 对于OHLC使用前向填充，然后后向填充
                merged_data[col] = merged_data[col].fillna(method='ffill').fillna(method='bfill')
            
            # 确认是否所有NaN都已处理
            nan_count_after = merged_data.isna().sum().sum()
            print(f"填充后剩余NaN值: {nan_count_after}个")
            
            # 如果仍有NaN值，则删除对应行
            if nan_count_after > 0:
                original_size = len(merged_data)
                merged_data = merged_data.dropna()
                print(f"删除了{original_size - len(merged_data)}行含有NaN值的数据")
        
        # 检查是否有较大的时间间隔
        time_diffs = merged_data.index.to_series().diff().dt.total_seconds()
        large_gaps = time_diffs[time_diffs > 3600]  # 超过1小时的间隔
        if not large_gaps.empty:
            print(f"警告: 数据中有{len(large_gaps)}个大于1小时的时间间隔")
            print("前5个大间隔:")
            for ts, gap in large_gaps.head().items():
                print(f"时间: {ts}, 间隔: {gap/3600:.2f}小时")
        
        # 分析信号变化
        signal_changes = merged_data['Signal'].diff().ne(0).sum()
        print(f"数据中的信号变化次数: {signal_changes}")
        print(f"信号分布: {merged_data['Signal'].value_counts()}")
        
        # 确保数据类型正确
        for col in required_columns:
            merged_data[col] = merged_data[col].astype(float)
        
        # 按周分析信号分布
        weekly_signal = merged_data['Signal'].resample('W').count()
        print("\n按周统计的信号数量:")
        print(weekly_signal.head(10))  # 显示前10周
        
        return merged_data
        
    except Exception as e:
        print(f"数据准备过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主回测函数"""
    # 配置参数
    data_file = 'minute_data.csv'  # 分钟数据文件
    signal_file = 'predictions_with_dynamic_threshold.csv'  # 信号文件
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 分钟数据文件 '{data_file}' 不存在!")
        available_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
        if available_csvs:
            print(f"目录中可用的CSV文件: {', '.join(available_csvs)}")
            print(f"请修改脚本中的 data_file 参数为正确的文件名")
        return None
    
    if not os.path.exists(signal_file):
        print(f"错误: 信号文件 '{signal_file}' 不存在!")
        available_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
        if available_csvs:
            print(f"目录中可用的CSV文件: {', '.join(available_csvs)}")
            print(f"请修改脚本中的 signal_file 参数为正确的文件名")
        return None
    
    # 准备数据
    try:
        data = prepare_data(data_file, signal_file)
        print(f"数据加载完成，共 {len(data)} 个数据点")
        
        # 降低采样频率 - 改为5分钟K线，减少交易频率
        data = data.resample('5min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Signal': 'last'  # 使用每5分钟最后一个信号
        }).dropna()
        print(f"降采样后的数据点数: {len(data)}")
        
        # 配置回测参数 - 增加初始资金
        initial_capital = 5000000  # 初始资金500万
        commission = 0.0003  # 交易所手续费率，0.03%
        
        # 测试不同的参数组合
        results_all = {}
        
        # 策略参数组1 - 标准设置
        strategy_params1 = {
            'use_stoploss': True,
            'sl_percent': 0.008,       # 0.8%止损
            'use_takeprofit': True,
            'tp_percent': 0.02,        # 2%止盈
            'position_size': 0.05,     # 每次只用5%资金
            'max_daily_loss': 0.01,    # 日内亏损1%停止交易
            'cool_down_hours': 4,      # 亏损后休息4小时
            'confirm_threshold': 1,    # 需要1个信号确认
            'require_reverse_signal': False  # 不要求反向信号
        }
        
        # 策略参数组2 - 频繁交易
        strategy_params2 = {
            'use_stoploss': True,
            'sl_percent': 0.01,        # 1%止损
            'use_takeprofit': True,
            'tp_percent': 0.015,       # 1.5%止盈
            'position_size': 0.03,     # 每次只用3%资金
            'max_daily_loss': 0.015,   # 日内亏损1.5%停止交易
            'cool_down_hours': 2,      # 亏损后休息2小时
            'confirm_threshold': 1,    # 需要1个信号确认
            'require_reverse_signal': False  # 不要求反向信号
        }
        
        # 策略参数组3 - 保守策略(要求反向信号)
        strategy_params3 = {
            'use_stoploss': True,
            'sl_percent': 0.008,       # 0.8%止损
            'use_takeprofit': True,
            'tp_percent': 0.02,        # 2%止盈
            'position_size': 0.05,     # 每次只用5%资金
            'max_daily_loss': 0.01,    # 日内亏损1%停止交易
            'cool_down_hours': 4,      # 亏损后休息4小时
            'confirm_threshold': 1,    # 需要1个信号确认
            'require_reverse_signal': True  # 要求反向信号
        }
        
        # 策略参数组4 - 激进策略(更高频交易)
        strategy_params4 = {
            'use_stoploss': True,
            'sl_percent': 0.012,       # 1.2%止损
            'use_takeprofit': True,
            'tp_percent': 0.01,        # 1.0%止盈，更快获利了结
            'position_size': 0.02,     # 每次只用2%资金
            'max_daily_loss': 0.02,    # 日内亏损2%停止交易
            'cool_down_hours': 1,      # 亏损后休息1小时
            'confirm_threshold': 1,    # 需要1个信号确认
            'require_reverse_signal': False, # 不要求反向信号
            'max_long_positions': 3,   # 最大多头持仓数量为3
            'max_short_positions': 3   # 最大空头持仓数量为3
        }
        
        # 运行回测 - 标准设置
        print("\n====== 回测1: 标准设置 ======")
        bt1 = Backtest(data, SignalStrategy, 
                      cash=initial_capital, 
                      commission=commission,
                      margin=0.15,
                      exclusive_orders=True,
                      trade_on_close=False)
        
        results1 = bt1.run(**strategy_params1)
        results_all['标准设置'] = results1
        
        print(f"总回报率: {results1['Return [%]']:.2f}%")
        print(f"交易次数: {results1['# Trades']}")
        print(f"夏普比率: {results1['Sharpe Ratio']:.4f}")
        
        bt1.plot(filename="backtest_results1.html", open_browser=False, 
                resample='1h')
        
        # 运行回测 - 频繁交易
        print("\n====== 回测2: 频繁交易 ======")
        bt2 = Backtest(data, SignalStrategy, 
                      cash=initial_capital, 
                      commission=commission,
                      margin=0.15,
                      exclusive_orders=True,
                      trade_on_close=False)
        
        results2 = bt2.run(**strategy_params2)
        results_all['频繁交易'] = results2
        
        print(f"总回报率: {results2['Return [%]']:.2f}%")
        print(f"交易次数: {results2['# Trades']}")
        print(f"夏普比率: {results2['Sharpe Ratio']:.4f}")
        
        bt2.plot(filename="backtest_results2.html", open_browser=False, 
                resample='1h')
        
        # 运行回测 - 保守策略
        print("\n====== 回测3: 保守策略(要求反向信号) ======")
        bt3 = Backtest(data, SignalStrategy, 
                      cash=initial_capital, 
                      commission=commission,
                      margin=0.15,
                      exclusive_orders=True,
                      trade_on_close=False)
        
        results3 = bt3.run(**strategy_params3)
        results_all['保守策略'] = results3
        
        print(f"总回报率: {results3['Return [%]']:.2f}%")
        print(f"交易次数: {results3['# Trades']}")
        print(f"夏普比率: {results3['Sharpe Ratio']:.4f}")
        
        bt3.plot(filename="backtest_results3.html", open_browser=False, 
                resample='1h')
        
        # 运行回测 - 激进策略
        print("\n====== 回测4: 激进策略(更高频交易) ======")
        bt4 = Backtest(data, SignalStrategy, 
                      cash=initial_capital, 
                      commission=commission,
                      margin=0.15,
                      exclusive_orders=True,
                      trade_on_close=False)
        
        results4 = bt4.run(**strategy_params4)
        results_all['激进策略'] = results4
        
        print(f"总回报率: {results4['Return [%]']:.2f}%")
        print(f"交易次数: {results4['# Trades']}")
        print(f"夏普比率: {results4['Sharpe Ratio']:.4f}")
        
        # 输出比较结果
        print("\n====== 三种策略比较 ======")
        comparison = pd.DataFrame({
            '标准设置': [
                results1['Return [%]'],
                results1['# Trades'],
                results1['Sharpe Ratio'],
                results1['Max. Drawdown [%]'],
                results1['Win Rate [%]']
            ],
            '频繁交易': [
                results2['Return [%]'],
                results2['# Trades'],
                results2['Sharpe Ratio'],
                results2['Max. Drawdown [%]'],
                results2['Win Rate [%]']
            ],
            '保守策略': [
                results3['Return [%]'],
                results3['# Trades'],
                results3['Sharpe Ratio'],
                results3['Max. Drawdown [%]'],
                results3['Win Rate [%]']
            ]
        }, index=['回报率(%)', '交易次数', '夏普比率', '最大回撤(%)', '胜率(%)'])
        
        print(comparison)
        
        # 找出交易次数最多且夏普比率最高的策略
        best_sharpe = max(results_all.items(), key=lambda x: x[1]['Sharpe Ratio'])[0]
        most_trades = max(results_all.items(), key=lambda x: x[1]['# Trades'])[0]
        
        print(f"\n夏普比率最高的策略: {best_sharpe}")
        print(f"交易次数最多的策略: {most_trades}")
        
        # 打开表现最好的策略结果
        best_strategy = best_sharpe
        if best_strategy == '标准设置':
            best_results_file = "backtest_results1.html"
        elif best_strategy == '频繁交易':
            best_results_file = "backtest_results2.html"
        else:
            best_results_file = "backtest_results3.html"
        
        # 打开最佳策略的图表
        import webbrowser
        webbrowser.open(best_results_file)
        
        print(f"\n回测完成，最佳策略结果已保存至 {best_results_file}")
        
        return results_all
        
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查数据文件格式是否正确，或调整策略参数后重试")
        return None


if __name__ == "__main__":
    main()
