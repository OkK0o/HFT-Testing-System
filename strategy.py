import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Position(Enum):
    """持仓状态枚举"""
    LONG = 1   # 多头
    SHORT = -1 # 空头
    FLAT = 0   # 空仓

@dataclass
class TradeRecord:
    """交易记录"""
    entry_time: pd.Timestamp    # 开仓时间
    exit_time: pd.Timestamp     # 平仓时间
    position: Position          # 持仓方向
    entry_price: float         # 开仓价格
    exit_price: float          # 平仓价格
    pnl: float                # 交易盈亏
    holding_periods: int      # 持仓周期数

class Strategy:
    """交易策略类"""
    
    def __init__(self,
                 signal_threshold: float = 0.0,      # 开仓信号阈值
                 stop_loss: float = 0.02,           # 止损比例
                 take_profit: float = 0.05,         # 止盈比例
                 max_holding_periods: int = 20,     # 最大持仓周期
                 min_holding_periods: int = 1,      # 最小持仓周期
                 commission_rate: float = 0.0003,   # 手续费率
                 size: int = 1                      # 每次交易手数
                 ):
        """
        初始化策略
        
        Args:
            signal_threshold: 开仓信号阈值
            stop_loss: 止损比例
            take_profit: 止盈比例
            max_holding_periods: 最大持仓周期
            min_holding_periods: 最小持仓周期
            commission_rate: 手续费率
            size: 每次交易手数
        """
        self.signal_threshold = signal_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_periods = max_holding_periods
        self.min_holding_periods = min_holding_periods
        self.commission_rate = commission_rate
        self.size = size
        
        # 交易记录
        self.trades: List[TradeRecord] = []
        
        # 当前持仓状态
        self.current_position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.holding_periods = 0
    
    def _should_open_long(self, signal: float) -> bool:
        """判断是否应该开多"""
        return (signal > self.signal_threshold and 
                self.current_position == Position.FLAT)
    
    def _should_open_short(self, signal: float) -> bool:
        """判断是否应该开空"""
        return (signal < -self.signal_threshold and 
                self.current_position == Position.FLAT)
    
    def _should_close_position(self, 
                             current_price: float, 
                             current_time: pd.Timestamp) -> bool:
        """
        判断是否应该平仓
        
        Args:
            current_price: 当前价格
            current_time: 当前时间
            
        Returns:
            bool: 是否应该平仓
        """
        if self.current_position == Position.FLAT:
            return False
            
        returns = ((current_price - self.entry_price) / self.entry_price * 
                  self.current_position.value)
        
        if returns <= -self.stop_loss or returns >= self.take_profit:
            return True
        
        if self.holding_periods >= self.max_holding_periods:
            return True
            
        return False
    
    def backtest(self,
                df: pd.DataFrame,
                price_col: str = 'mid_price',
                signal_col: str = 'predicted_signal',
                datetime_col: str = 'DateTime') -> Tuple[pd.DataFrame, Dict]:
        """
        回测策略
        
        Args:
            df: 输入数据，包含价格和信号
            price_col: 价格列名
            signal_col: 信号列名
            datetime_col: 时间列名
            
        Returns:
            positions_df: 包含持仓信息的DataFrame
            metrics: 回测指标字典
        """
        # 初始化结果
        positions = []
        self.trades = []
        self.current_position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.holding_periods = 0
        
        for i, row in df.iterrows():
            current_time = row[datetime_col]
            current_price = row[price_col]
            signal = row[signal_col]
            
            if self.current_position != Position.FLAT:
                self.holding_periods += 1
            
            if (self.current_position != Position.FLAT and 
                self.holding_periods >= self.min_holding_periods and
                self._should_close_position(current_price, current_time)):
                pnl = ((current_price - self.entry_price) * 
                       self.current_position.value * 
                       self.size)
                pnl -= (self.entry_price + current_price) * self.commission_rate * self.size
                
                self.trades.append(TradeRecord(
                    entry_time=self.entry_time,
                    exit_time=current_time,
                    position=self.current_position,
                    entry_price=self.entry_price,
                    exit_price=current_price,
                    pnl=pnl,
                    holding_periods=self.holding_periods
                ))
                
                self.current_position = Position.FLAT
                self.entry_price = 0.0
                self.entry_time = None
                self.holding_periods = 0
            
            elif self.current_position == Position.FLAT:
                if self._should_open_long(signal):
                    self.current_position = Position.LONG
                    self.entry_price = current_price
                    self.entry_time = current_time
                    self.holding_periods = 0
                elif self._should_open_short(signal):
                    self.current_position = Position.SHORT
                    self.entry_price = current_price
                    self.entry_time = current_time
                    self.holding_periods = 0
            
            positions.append({
                'datetime': current_time,
                'price': current_price,
                'signal': signal,
                'position': self.current_position.value,
                'holding_periods': self.holding_periods
            })
        
        positions_df = pd.DataFrame(positions)
        
        metrics = self._calculate_metrics(positions_df)
        
        return positions_df, metrics
    
    def _calculate_metrics(self, positions_df: pd.DataFrame) -> Dict:
        """
        计算回测指标
        
        Args:
            positions_df: 持仓信息DataFrame
            
        Returns:
            包含回测指标的字典
        """
        metrics = {}
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'long_rate': 0.0,
                'short_rate': 0.0
            }
        
        metrics['total_trades'] = len(self.trades)
        metrics['win_rate'] = len([t for t in self.trades if t.pnl > 0]) / len(self.trades)
        metrics['total_pnl'] = sum(t.pnl for t in self.trades)
        returns = [(t.exit_price - t.entry_price) / t.entry_price * t.position.value 
                  for t in self.trades]
        if len(returns) > 1:
            annual_factor = 252  # 交易日数量
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            metrics['sharpe_ratio'] = np.sqrt(annual_factor) * avg_return / std_return if std_return != 0 else 0
        else:
            metrics['sharpe_ratio'] = 0.0
        
        cumulative_returns = np.cumsum(returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - rolling_max
        metrics['max_drawdown'] = abs(min(drawdowns)) if drawdowns.size > 0 else 0
        
        long_trades = [t for t in self.trades if t.position == Position.LONG]
        short_trades = [t for t in self.trades if t.position == Position.SHORT]
        
        metrics['long_rate'] = len([t for t in self.trades if t.position == Position.LONG]) / len(self.trades)
        metrics['short_rate'] = len([t for t in self.trades if t.position == Position.SHORT]) / len(self.trades)
        metrics['avg_holding_periods'] = np.mean([t.holding_periods for t in self.trades])
        
        profits = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.trades if t.pnl < 0]
        metrics['profit_loss_ratio'] = (np.mean(profits) / np.mean(losses) 
                                      if losses and profits else 0.0)
        
        return metrics
    
    def plot_results(self, positions_df: pd.DataFrame) -> None:
        """
        绘制回测结果
        
        Args:
            positions_df: 持仓信息DataFrame
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        ax1.plot(positions_df['datetime'], positions_df['price'], label='Price')
        ax1.scatter(positions_df[positions_df['position'] == 1]['datetime'],
                   positions_df[positions_df['position'] == 1]['price'],
                   color='g', marker='^', label='Long')
        ax1.scatter(positions_df[positions_df['position'] == -1]['datetime'],
                   positions_df[positions_df['position'] == -1]['price'],
                   color='r', marker='v', label='Short')
        ax1.set_title('Price and Positions')
        ax1.legend()
        
        ax2.plot(positions_df['datetime'], positions_df['signal'], label='Signal')
        ax2.axhline(y=self.signal_threshold, color='g', linestyle='--', 
                    label='Long Threshold')
        ax2.axhline(y=-self.signal_threshold, color='r', linestyle='--',
                    label='Short Threshold')
        ax2.set_title('Trading Signal')
        ax2.legend()
        
        # 绘制累计收益
        if self.trades:
            trade_times = [t.exit_time for t in self.trades]
            cumulative_pnl = np.cumsum([t.pnl for t in self.trades])
            ax3.plot(trade_times, cumulative_pnl, label='Cumulative P&L')
            ax3.set_title('Cumulative P&L')
            ax3.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df = pd.read_feather("data.feather")
    
    strategy = Strategy(
        signal_threshold=0.5,
        stop_loss=0.02,
        take_profit=0.05,
        max_holding_periods=20,
        min_holding_periods=1,
        commission_rate=0.0003
    )
    
    positions_df, metrics = strategy.backtest(df)
    print("\n=== 回测结果 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    strategy.plot_results(positions_df)