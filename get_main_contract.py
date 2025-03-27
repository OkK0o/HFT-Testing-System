import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def get_contracts_info(if_folder, start_year=2022, end_year=2025):
    """获取指定年份范围内的所有合约基本信息"""
    folders = [f for f in os.listdir(if_folder) if os.path.isdir(os.path.join(if_folder, f)) and f.startswith('IF')]
    
    contract_info = []
    pattern = r'IF(\d{2})(\d{2})'
    
    for folder in folders:
        match = re.match(pattern, folder)
        if match:
            year, month = match.groups()
            full_year = int("20" + year)
            month = int(month)
            
            # 过滤年份范围
            if full_year < start_year or full_year > end_year:
                continue
                
            # 构建合约到期日 (假设每月第三个周五)
            first_day = datetime(full_year, month, 1)
            weekday = first_day.weekday()
            days_until_first_friday = (4 - weekday) % 7
            third_friday = first_day + timedelta(days=days_until_first_friday + 14)
            
            # 检查文件夹内文件数量
            folder_path = os.path.join(if_folder, folder)
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.feather')])
            
            contract_info.append({
                'folder': folder,
                'year': full_year,
                'month': month,
                'expiry_date': third_friday,
                'is_quarterly': month in [3, 6, 9, 12],
                'file_count': file_count
            })
    
    # 按到期日排序
    contract_info.sort(key=lambda x: (x['year'], x['month']))
    return contract_info

def load_daily_volume_data(if_folder, contract_info, start_date=datetime(2022, 1, 1), end_date=datetime(2025, 12, 31)):
    """加载指定日期范围内每个合约每天的成交量数据"""
    date_pattern = r'(\d{8})\.feather'
    daily_volumes = {}  # {date: {contract: volume}}
    contract_dates = {}  # {contract: [dates]}
    
    for contract in contract_info:
        folder_path = os.path.join(if_folder, contract['folder'])
        contract_dates[contract['folder']] = []
        
        for file in os.listdir(folder_path):
            match = re.match(date_pattern, file)
            if match:
                date_str = match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
                
                # 过滤日期范围
                if date < start_date or date > end_date:
                    continue
                    
                contract_dates[contract['folder']].append(date)
                
                try:
                    # 读取文件计算总成交量
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_feather(file_path)
                    
                    if 'Volume' in df.columns:
                        total_volume = df['Volume'].sum()
                    else:
                        # 尝试其他可能的成交量列名
                        volume_columns = ['volume', 'VOL', 'vol', 'VOLUME']
                        for col in volume_columns:
                            if col in df.columns:
                                total_volume = df[col].sum()
                                break
                        else:
                            total_volume = 0  # 找不到成交量列
                    
                    if date not in daily_volumes:
                        daily_volumes[date] = {}
                    daily_volumes[date][contract['folder']] = total_volume
                    
                except Exception as e:
                    print(f"读取 {file_path} 时出错: {str(e)}")
    
    return daily_volumes, contract_dates

def get_main_contract_by_volume(daily_volumes, window_size=5, min_volume_ratio=0.3):
    """基于N日成交量均值确定主力合约
    
    Args:
        daily_volumes: 每日成交量字典 {date: {contract: volume}}
        window_size: 计算均值的天数窗口
        min_volume_ratio: 主力合约最小成交量占比
        
    Returns:
        主力合约字典 {date: contract}
    """
    main_contracts = {}
    
    # 转换为DataFrame便于计算滚动均值
    dates = sorted(daily_volumes.keys())
    contracts = set()
    for date_volumes in daily_volumes.values():
        contracts.update(date_volumes.keys())
    
    # 创建日期-合约-成交量的DataFrame
    volume_data = []
    for date in dates:
        for contract in contracts:
            volume = daily_volumes.get(date, {}).get(contract, 0)
            volume_data.append({'date': date, 'contract': contract, 'volume': volume})
    
    volume_df = pd.DataFrame(volume_data)
    
    # 计算每个合约的滚动成交量均值
    rolling_volumes = {}
    for contract in contracts:
        contract_volumes = volume_df[volume_df['contract'] == contract].sort_values('date')
        contract_volumes['rolling_avg'] = contract_volumes['volume'].rolling(window=window_size, min_periods=1).mean()
        
        for _, row in contract_volumes.iterrows():
            date = row['date']
            if date not in rolling_volumes:
                rolling_volumes[date] = {}
            rolling_volumes[date][contract] = row['rolling_avg']
    
    # 对每个日期，选择滚动均值最大的合约
    for date in dates:
        if date not in rolling_volumes or not rolling_volumes[date]:
            continue
        
        # 找出当日滚动均值最大的合约
        date_volumes = rolling_volumes[date]
        max_volume_contract = max(date_volumes.items(), key=lambda x: x[1])
        
        # 确保成交量占比足够大
        total_volume = sum(date_volumes.values())
        if total_volume > 0 and max_volume_contract[1] / total_volume >= min_volume_ratio:
            main_contracts[date] = max_volume_contract[0]
    
    return main_contracts

def get_main_contract_by_month_rule(contract_info, contract_dates):
    """基于固定月度规则确定主力合约"""
    main_contracts = {}
    all_dates = set()
    
    # 收集所有日期
    for dates in contract_dates.values():
        all_dates.update(dates)
    
    # 按日期排序
    all_dates = sorted(all_dates)
    
    # 确定每个日期的主力合约 (使用固定月度规则)
    current_year = None
    current_main_contracts = {}  # 每年的主力合约映射
    
    # 预定义每年的主力合约月份顺序
    month_sequence = [3, 6, 9, 12, 3, 6, 9, 12]
    
    for date in all_dates:
        if current_year != date.year:
            current_year = date.year
            # 为当年确定主力合约顺序
            current_main_contracts = {}
            
            # 找出当年存在的季月合约
            year_contracts = [c for c in contract_info if c['year'] == current_year and c['is_quarterly']]
            
            # 按月份排序
            year_contracts.sort(key=lambda x: x['month'])
            
            # 如果这一年的季度合约不足，可能需要考虑下一年的合约
            if len(year_contracts) < 4:
                next_year_contracts = [c for c in contract_info if c['year'] == current_year + 1 and c['is_quarterly']]
                next_year_contracts.sort(key=lambda x: x['month'])
                year_contracts.extend(next_year_contracts)
            
            # 为每个月设置对应的主力合约
            for i in range(1, 13):
                # 对于每个月份，选择下一个到期的季度合约
                next_quarterly = None
                for contract in year_contracts:
                    if contract['month'] > i or (contract['year'] > current_year and i > 9):
                        next_quarterly = contract
                        break
                
                if next_quarterly:
                    current_main_contracts[i] = next_quarterly['folder']
                elif year_contracts:
                    # 如果没有找到下一个季度合约，使用最后一个
                    current_main_contracts[i] = year_contracts[-1]['folder']
        
        # 使用当月对应的主力合约
        month = date.month
        if month in current_main_contracts:
            main_contracts[date] = current_main_contracts[month]
    
    return main_contracts

def apply_continuous_rollover(main_contracts_by_date, if_folder, rollover_method='price_ratio', batch_size=30):
    """分批处理数据以减少内存使用"""
    continuous_data = []
    all_dates = sorted(main_contracts_by_date.keys())
    previous_contract = None
    adjustment_factor = 1.0  # 价格调整因子
    
    # 分批处理日期
    for batch_start in range(0, len(all_dates), batch_size):
        batch_dates = all_dates[batch_start:batch_start + batch_size]
        batch_data = []
        
        for date in batch_dates:
            contract = main_contracts_by_date[date]
            date_str = date.strftime('%Y%m%d')
            file_path = os.path.join(if_folder, contract, f"{date_str}.feather")
            
            if not os.path.exists(file_path):
                continue
                
            try:
                # 读取数据（只加载必要的列以减少内存使用）
                essential_columns = None  # 设置为None表示加载所有列，可以指定列名列表
                if essential_columns:
                    df = pd.read_feather(file_path, columns=essential_columns)
                else:
                    df = pd.read_feather(file_path)
                
                # 添加合约信息
                df['contract_id'] = contract
                df['date'] = date
                
                # 标记换约
                is_rollover = previous_contract is not None and previous_contract != contract
                df['contract_switch'] = 1 if is_rollover else 0
                
                # 处理连续合约的价格调整
                if is_rollover and rollover_method != 'none':
                    # 查找上一个合约在换约日的收盘价
                    prev_file = os.path.join(if_folder, previous_contract, f"{date_str}.feather")
                    if os.path.exists(prev_file):
                        prev_df = pd.read_feather(prev_file)
                        
                        # 获取收盘价
                        if 'ClosePrice' in df.columns and 'ClosePrice' in prev_df.columns:
                            curr_close = df['ClosePrice'].iloc[0] if not df.empty else None
                            prev_close = prev_df['ClosePrice'].iloc[0] if not prev_df.empty else None
                            
                            if curr_close is not None and prev_close is not None:
                                if rollover_method == 'price_ratio':
                                    # 价格比例调整
                                    new_factor = adjustment_factor * (prev_close / curr_close)
                                    # 应用调整因子到价格列
                                    for col in ['LastPrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']:
                                        if col in df.columns:
                                            df[col] = df[col] * new_factor
                                    adjustment_factor = new_factor
                                
                                elif rollover_method == 'price_diff':
                                    # 价格差异调整
                                    diff = (adjustment_factor * prev_close) - curr_close
                                    # 应用调整到价格列
                                    for col in ['LastPrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']:
                                        if col in df.columns:
                                            df[col] = df[col] + diff
                
                batch_data.append(df)
                previous_contract = contract
                
                print(f"处理 {date_str} 数据，主力合约: {contract}{' (换约)' if is_rollover else ''}")
                
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {str(e)}")
        
        # 将此批次数据合并并添加到结果中
        if batch_data:
            try:
                batch_df = pd.concat(batch_data, ignore_index=True)
                continuous_data.append(batch_df)
                
                # 强制释放内存
                del batch_data
                import gc
                gc.collect()
                print(f"完成批次 {batch_start//batch_size + 1}/{(len(all_dates) + batch_size - 1)//batch_size}，释放内存")
                
            except Exception as e:
                print(f"合并批次数据时出错: {str(e)}")
    
    # 拼接所有批次数据
    if continuous_data:
        try:
            print("合并所有批次数据...")
            result_df = pd.concat(continuous_data, ignore_index=True)
            
            # 释放内存
            del continuous_data
            gc.collect()
            
            return result_df
        except Exception as e:
            print(f"合并所有批次数据时出错: {str(e)}")
            # 如果最终合并失败，返回最后一个批次的数据
            if continuous_data:
                return continuous_data[-1]
            else:
                return pd.DataFrame()
    else:
        return pd.DataFrame()

def visualize_main_contracts(main_contracts, contract_dates, output_file='main_contracts.png'):
    """可视化主力合约变化"""
    # 创建日期-合约映射表
    date_contract_df = []
    
    for date, contract in sorted(main_contracts.items()):
        date_contract_df.append({'date': date, 'contract': contract})
    
    df = pd.DataFrame(date_contract_df)
    
    if df.empty:
        print("没有足够的数据进行可视化")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 为每个合约分配一个唯一的数字ID用于绘图
    unique_contracts = df['contract'].unique()
    contract_ids = {contract: i for i, contract in enumerate(unique_contracts)}
    
    df['contract_id'] = df['contract'].map(contract_ids)
    
    # 绘制主力合约变化
    plt.plot(df['date'], df['contract_id'], 'o-', linewidth=2, markersize=5)
    
    # 设置y轴标签为合约代码
    plt.yticks(range(len(unique_contracts)), unique_contracts)
    
    # 标记换月点
    rollovers = df[df['contract'].shift() != df['contract']].copy()
    if not rollovers.empty:
        plt.plot(rollovers['date'], rollovers['contract_id'], 'ro', markersize=8)
    
    plt.title('主力合约变化图')
    plt.xlabel('日期')
    plt.ylabel('合约代码')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file)
    plt.close()
    print(f"主力合约变化图已保存至 {output_file}")

def visualize_volume_and_contracts(daily_volumes, main_contracts, window_size, output_file='volume_contracts.png'):
    """可视化成交量与主力合约关系"""
    # 准备数据
    all_dates = sorted(daily_volumes.keys())
    all_contracts = set()
    for date_volumes in daily_volumes.values():
        all_contracts.update(date_volumes.keys())
    
    # 创建成交量数据框
    volume_data = []
    for date in all_dates:
        for contract in all_contracts:
            volume = daily_volumes.get(date, {}).get(contract, 0)
            is_main = main_contracts.get(date) == contract
            volume_data.append({
                'date': date, 
                'contract': contract, 
                'volume': volume,
                'is_main': is_main
            })
    
    df = pd.DataFrame(volume_data)
    
    # 计算滚动均值
    for contract in all_contracts:
        contract_df = df[df['contract'] == contract].sort_values('date')
        if not contract_df.empty:
            df.loc[df['contract'] == contract, 'rolling_volume'] = contract_df['volume'].rolling(window=window_size, min_periods=1).mean()
    
    # 选择图表展示的合约（成交量最大的前5个）
    top_contracts = df.groupby('contract')['volume'].sum().nlargest(5).index.tolist()
    
    plt.figure(figsize=(15, 8))
    
    # 绘制每个主要合约的滚动均值成交量
    for contract in top_contracts:
        contract_df = df[(df['contract'] == contract) & (df['rolling_volume'] > 0)].sort_values('date')
        plt.plot(contract_df['date'], contract_df['rolling_volume'], 
                 label=contract, alpha=0.7, linewidth=2)
    
    # 标记主力合约点
    main_df = df[df['is_main']].sort_values('date')
    plt.scatter(main_df['date'], main_df['rolling_volume'], 
                color='red', s=50, alpha=0.7, label='主力合约')
    
    plt.title(f'{window_size}日滚动成交量均值与主力合约')
    plt.xlabel('日期')
    plt.ylabel('成交量滚动均值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file)
    plt.close()
    print(f"成交量与主力合约关系图已保存至 {output_file}")

def main():
    if_folder = "IF"  # 修改为您的文件夹路径
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    # 设置年份和日期范围参数
    start_year = 2022
    end_year = 2025
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # 滚动成交量均值的窗口大小
    volume_window_sizes = [3, 5]  # 可以测试多个窗口大小
    
    print(f"获取{start_year}年至{end_year}年的合约信息...")
    contract_info = get_contracts_info(if_folder, start_year, end_year)
    print(f"共找到 {len(contract_info)} 个合约")
    
    print(f"加载{start_date.strftime('%Y-%m-%d')}至{end_date.strftime('%Y-%m-%d')}的成交量数据...")
    daily_volumes, contract_dates = load_daily_volume_data(if_folder, contract_info, start_date, end_date)
    print(f"共加载 {len(daily_volumes)} 个交易日的数据")
    
    # 监控内存使用
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
        print(f"当前内存使用: {memory_mb:.2f} MB")
    except ImportError:
        print("psutil模块未安装，无法监控内存使用")
    
    # 测试不同窗口大小的成交量均值方法
    for window_size in volume_window_sizes:
        print(f"\n方法1: 基于{window_size}日成交量均值识别主力合约")
        volume_main_contracts = get_main_contract_by_volume(daily_volumes, window_size=window_size)
        print(f"基于{window_size}日成交量均值识别出 {len(volume_main_contracts)} 个交易日的主力合约")
        
        # 可选：可视化成交量主力合约（可以注释掉以节省内存）
        visualize_main_contracts(volume_main_contracts, contract_dates, 
                               os.path.join(output_folder, f"volume{window_size}d_main_contracts.png"))
        
        # 创建连续合约数据（使用批处理）
        print(f"\n使用{window_size}日成交量均值方法创建连续合约...")
        continuous_df = apply_continuous_rollover(volume_main_contracts, if_folder, 
                                               rollover_method='price_ratio', batch_size=30)
        
        if not continuous_df.empty:
            # 保存结果
            output_file = os.path.join(output_folder, f"continuous_IF_volume{window_size}d.feather")
            continuous_df.to_feather(output_file)
            print(f"连续合约数据已保存至 {output_file}")
            print(f"数据形状: {continuous_df.shape}")
            print(f"时间范围: {continuous_df['date'].min()} 至 {continuous_df['date'].max()}")
            print(f"包含换约点数量: {continuous_df['contract_switch'].sum()}")
            
            # 保存CSV汇总便于查看
            csv_file = os.path.join(output_folder, f"continuous_IF_volume{window_size}d_summary.csv")
            summary_df = continuous_df.groupby(['date', 'contract_id'])['contract_switch'].first().reset_index()
            summary_df.to_csv(csv_file, index=False)
            print(f"连续合约汇总已保存至 {csv_file}")
            
            # 释放内存
            del continuous_df
            import gc
            gc.collect()
    
    # 基于月度规则的方法
    print("\n方法2: 基于月度规则识别主力合约")
    month_main_contracts = get_main_contract_by_month_rule(contract_info, contract_dates)
    print(f"基于月度规则识别出 {len(month_main_contracts)} 个交易日的主力合约")
    
    # 可选：可视化月度规则主力合约
    visualize_main_contracts(month_main_contracts, contract_dates, 
                           os.path.join(output_folder, "month_main_contracts.png"))
    
    # 创建月度规则连续合约数据（使用批处理）
    print("\n使用月度规则方法创建连续合约...")
    continuous_df = apply_continuous_rollover(month_main_contracts, if_folder, 
                                           rollover_method='price_ratio', batch_size=30)
    
    if not continuous_df.empty:
        # 保存结果
        output_file = os.path.join(output_folder, "continuous_IF_month.feather")
        continuous_df.to_feather(output_file)
        print(f"连续合约数据已保存至 {output_file}")
        print(f"数据形状: {continuous_df.shape}")
        print(f"时间范围: {continuous_df['date'].min()} 至 {continuous_df['date'].max()}")
        print(f"包含换约点数量: {continuous_df['contract_switch'].sum()}")
        
        # 保存CSV汇总
        csv_file = os.path.join(output_folder, "continuous_IF_month_summary.csv")
        summary_df = continuous_df.groupby(['date', 'contract_id'])['contract_switch'].first().reset_index()
        summary_df.to_csv(csv_file, index=False)
        print(f"连续合约汇总已保存至 {csv_file}")

if __name__ == "__main__":
    main()