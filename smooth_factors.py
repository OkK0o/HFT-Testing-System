import pandas as pd
import numpy as np
import time
import os
import gc
import argparse
from factor_manager import FactorManager, FactorFrequency
from factor_register import register_all_factors
from factor_compute import smooth_factors

def smooth_factor_process(df, factor_cols=None, window=5, output_file=None, batch_size=5):
    """
    对指定的因子进行平滑处理
    
    Args:
        df: 输入的DataFrame
        factor_cols: 要平滑的因子列名列表，如果为None则使用所有非时间和ID列
        window: 平滑窗口大小
        output_file: 输出文件路径，如果为None则不保存结果
        batch_size: 批处理大小，避免内存溢出
        
    Returns:
        包含原始因子和平滑因子的DataFrame
    """
    start_time = time.time()
    
    # # 如果未指定因子列，尝试使用所有有效列
    # if factor_cols is None:
    #     # 排除常见的非因子列
    #     exclude_cols = ['InstruID', 'DateTime', 'TradDay', 'date', 'UpdateTime']
    #     exclude_cols += [col for col in df.columns if 'return' in col.lower()]
    #     factor_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 确保所有指定的因子都存在于DataFrame中
    missing_cols = [col for col in factor_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 以下列不存在于数据中: {missing_cols}")
        factor_cols = [col for col in factor_cols if col in df.columns]
    
    print(f"将对 {len(factor_cols)} 个因子应用EMA平滑 (窗口大小: {window})")
    
    # 注册所有因子（为了使用smooth_factors函数）
    register_all_factors()
    
    # 存储所有平滑因子名称
    all_smoothed_factors = []
    
    # 如果有很多因子，使用分批处理
    for i in range(0, len(factor_cols), batch_size):
        batch_factors = factor_cols[i:i+batch_size]
        print(f"处理第 {i//batch_size+1}/{(len(factor_cols)+batch_size-1)//batch_size} 批因子 ({len(batch_factors)}个)...")
        
        # 读取当前最新的数据
        if i == 0:
            current_df = df.copy()
        else:
            if 'temp_result' in locals():
                current_df = temp_result
            else:
                current_df = df.copy()
        
        # 应用平滑
        try:
            smoothed_df, smooth_factors_list = smooth_factors(
                current_df,
                factor_names=batch_factors,
                windows=window,
                methods='ema',
                register_factors=True
            )
            
            # 保存临时结果
            temp_result = smoothed_df
            
            # 添加平滑因子名称到列表
            all_smoothed_factors.extend(smooth_factors_list)
            
            print(f"完成 {len(smooth_factors_list)} 个因子的平滑")
            
            # 释放内存
            if i > 0:
                gc.collect()
                
        except Exception as e:
            print(f"处理因子批次时出错: {str(e)}")
    
    # 计算耗时
    duration = time.time() - start_time
    print(f"\n平滑处理完成! 耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
    print(f"总共生成了 {len(all_smoothed_factors)} 个平滑因子")
    
    # 显示部分平滑因子示例
    if all_smoothed_factors:
        print("示例平滑因子:")
        for factor in all_smoothed_factors[:5]:
            print(f"  - {factor}")
        if len(all_smoothed_factors) > 5:
            print(f"  - ... 等 {len(all_smoothed_factors)} 个因子")
    
    # 保存结果
    if output_file:
        print(f"\n保存结果到: {output_file}")
        if output_file.endswith('.feather'):
            temp_result.to_feather(output_file)
        elif output_file.endswith('.csv'):
            temp_result.to_csv(output_file, index=False)
        else:
            temp_result.to_feather(output_file)
    
    # 生成平滑因子名称列表
    smoothed_factor_info = pd.DataFrame({
        'original_factor': [f.split('_ema')[0] for f in all_smoothed_factors],
        'smoothed_factor': all_smoothed_factors,
        'window': window
    })
    
    if output_file:
        # 保存平滑因子列表
        output_dir = os.path.dirname(output_file)
        if not output_dir:
            output_dir = '.'
        
        factor_list_file = os.path.join(output_dir, "smoothed_factors_list.csv")
        smoothed_factor_info.to_csv(factor_list_file, index=False)
        print(f"平滑因子列表已保存至: {factor_list_file}")
    
    return temp_result, all_smoothed_factors, smoothed_factor_info

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='简化版因子平滑工具')
    parser.add_argument('--input', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output', type=str, help='输出数据文件路径')
    parser.add_argument('--window', type=int, default=5, help='平滑窗口大小')
    parser.add_argument('--batch-size', type=int, default=5, help='每批处理的因子数量')
    parser.add_argument('--factors', type=str, nargs='+', help='要平滑的因子列表（不指定则平滑所有有效列）')
    
    args = parser.parse_args()
    
    print(f"加载数据文件 {args.input}...")
    
    # 读取数据
    if args.input.endswith('.feather'):
        df = pd.read_feather(args.input)
    elif args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        raise ValueError(f"不支持的文件格式: {args.input}")
    
    print(f"数据形状: {df.shape}")
    
    # 执行平滑处理
    result_df, smoothed_factors, factor_info = smooth_factor_process(
        df=df,
        factor_cols=args.factors,
        window=args.window,
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    print("平滑处理完成!")

if __name__ == "__main__":
    main()