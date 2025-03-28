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

def apply_continuous_rollover(main_contracts_by_date, if_folder, output_folder, rollover_method='price_ratio'):
    """
    将主力合约数据保存到单独的文件夹中，不合并数据
    
    Args:
        main_contracts_by_date: 主力合约字典 {date: contract}
        if_folder: 原始数据文件夹
        output_folder: 输出文件夹
        rollover_method: 价格调整方法，可选 'none', 'price_ratio', 'price_diff'
    
    Returns:
        保存的文件数量
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    all_dates = sorted(main_contracts_by_date.keys())
    previous_contract = None
    adjustment_factor = 1.0  # 价格调整因子
    price_diff = 0.0  # 价格差异调整值
    processed_files = 0
    
    # 首先创建一个合约信息文件，记录每个日期对应的主力合约
    contract_info_df = []
    
    for date in all_dates:
        contract = main_contracts_by_date[date]
        date_str = date.strftime('%Y%m%d')
        
        # 检查是否存在换约
        is_rollover = previous_contract is not None and previous_contract != contract
        
        if is_rollover:
            # 获取换约日期的上一个合约价格和当前合约价格，用于价格调整
            prev_file = os.path.join(if_folder, previous_contract, f"{date_str}.feather")
            curr_file = os.path.join(if_folder, contract, f"{date_str}.feather")
            
            if os.path.exists(prev_file) and os.path.exists(curr_file) and rollover_method != 'none':
                try:
                    prev_df = pd.read_feather(prev_file)
                    curr_df = pd.read_feather(curr_file)
                    
                    # 获取收盘价
                    if 'ClosePrice' in prev_df.columns and 'ClosePrice' in curr_df.columns:
                        prev_close = prev_df['ClosePrice'].iloc[0] if not prev_df.empty else None
                        curr_close = curr_df['ClosePrice'].iloc[0] if not curr_df.empty else None
                        
                        if prev_close is not None and curr_close is not None:
                            if rollover_method == 'price_ratio':
                                # 价格比例调整
                                adjustment_factor = adjustment_factor * (prev_close / curr_close)
                            elif rollover_method == 'price_diff':
                                # 价格差异调整
                                price_diff = (price_diff + prev_close) - curr_close
                except Exception as e:
                    print(f"计算价格调整因子时出错: {str(e)}")
        
        # 记录主力合约信息
        contract_info_df.append({
            'date': date,
            'date_str': date_str,
            'contract': contract,
            'is_rollover': 1 if is_rollover else 0,
            'adjustment_factor': adjustment_factor,
            'price_diff': price_diff
        })
        
        # 更新前一个合约
        previous_contract = contract
    
    # 保存合约信息
    contract_info_df = pd.DataFrame(contract_info_df)
    contract_info_df.to_csv(os.path.join(output_folder, "main_contract_info.csv"), index=False)
    print(f"已保存主力合约信息到 {os.path.join(output_folder, 'main_contract_info.csv')}")
    
    # 处理并保存每个日期的数据
    for i, row in contract_info_df.iterrows():
        date = row['date']
        date_str = row['date_str']
        contract = row['contract']
        adjustment_factor = row['adjustment_factor']
        price_diff = row['price_diff']
        
        source_file = os.path.join(if_folder, contract, f"{date_str}.feather")
        target_file = os.path.join(output_folder, f"{date_str}.feather")
        
        if not os.path.exists(source_file):
            print(f"找不到源文件 {source_file}，跳过")
            continue
            
        try:
            # 读取数据
            df = pd.read_feather(source_file)
            
            # 添加合约信息
            df['contract_id'] = contract
            df['is_main'] = 1  # 标记为主力合约
            df['is_rollover'] = row['is_rollover']  # 标记换约
            
            # 处理价格调整
            if rollover_method != 'none':
                price_columns = ['LastPrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']
                available_columns = [col for col in price_columns if col in df.columns]
                
                if available_columns:
                    if rollover_method == 'price_ratio':
                        # 价格比例调整
                        for col in available_columns:
                            df[col] = df[col] * adjustment_factor
                    elif rollover_method == 'price_diff':
                        # 价格差异调整
                        for col in available_columns:
                            df[col] = df[col] + price_diff
            
            # 保存到目标文件
            df.to_feather(target_file)
            processed_files += 1
            
            print(f"处理 {date_str} 数据，主力合约: {contract}{' (换约)' if row['is_rollover'] else ''}")
            
        except Exception as e:
            print(f"处理文件 {source_file} 时出错: {str(e)}")
    
    print(f"共处理并保存了 {processed_files} 个文件到 {output_folder}")
    return processed_files

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
    
    # 为不同的主力合约选择方法创建不同的输出文件夹
    volume_output_folder = os.path.join(output_folder, "volume_main")
    month_output_folder = os.path.join(output_folder, "month_main")
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(volume_output_folder, exist_ok=True)
    os.makedirs(month_output_folder, exist_ok=True)
    
    # 设置年份和日期范围参数
    start_year = 2022
    end_year = 2025
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # 滚动成交量均值的窗口大小
    volume_window_size = 5
    
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
    
    # 基于成交量均值的方法
    print(f"\n方法1: 基于{volume_window_size}日成交量均值识别主力合约")
    volume_main_contracts = get_main_contract_by_volume(daily_volumes, window_size=volume_window_size)
    print(f"基于{volume_window_size}日成交量均值识别出 {len(volume_main_contracts)} 个交易日的主力合约")
    
    # 可视化成交量主力合约
    visualize_main_contracts(volume_main_contracts, contract_dates, 
                           os.path.join(output_folder, f"volume{volume_window_size}d_main_contracts.png"))
    visualize_volume_and_contracts(daily_volumes, volume_main_contracts, volume_window_size,
                                 os.path.join(output_folder, f"volume{volume_window_size}d_analysis.png"))
    
    # 保存成交量均值主力合约数据（按日期分别保存）
    print(f"\n将基于{volume_window_size}日成交量均值的主力合约数据保存至 {volume_output_folder}")
    processed_volume = apply_continuous_rollover(volume_main_contracts, if_folder, volume_output_folder, 
                                             rollover_method='price_ratio')
    
    # 基于月度规则的方法
    print("\n方法2: 基于月度规则识别主力合约")
    month_main_contracts = get_main_contract_by_month_rule(contract_info, contract_dates)
    print(f"基于月度规则识别出 {len(month_main_contracts)} 个交易日的主力合约")
    
    # 可视化月度规则主力合约
    visualize_main_contracts(month_main_contracts, contract_dates, 
                           os.path.join(output_folder, "month_main_contracts.png"))
    
    # 保存月度规则主力合约数据（按日期分别保存）
    print(f"\n将基于月度规则的主力合约数据保存至 {month_output_folder}")
    processed_month = apply_continuous_rollover(month_main_contracts, if_folder, month_output_folder, 
                                            rollover_method='price_ratio')
    
    print("\n处理完成!")
    print(f"基于{volume_window_size}日成交量均值: 处理了 {processed_volume} 个文件")
    print(f"基于月度规则: 处理了 {processed_month} 个文件")

if __name__ == "__main__":
    main()