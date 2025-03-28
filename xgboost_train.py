import pandas as pd
import numpy as np
import xgboost_signal
import factor_register
from factor_manager import FactorManager, FactorFrequency
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse
import pyarrow as pa
import pyarrow.feather as feather
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

def load_data_by_date(file_path, start_date=None, end_date=None):
    """按日期范围加载数据"""
    print("按日期范围加载数据...")
    reader = pa.ipc.RecordBatchFileReader(pa.memory_map(file_path))
    
    # 如果没有指定日期范围，使用所有数据
    if start_date is None and end_date is None:
        print("使用所有数据")
        start_date = '2022-01-01'  # 设置一个默认的开始日期
        end_date = '2024-12-31'    # 设置一个默认的结束日期
    
    # 转换日期格式
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    print(f"加载 {start_date} 至 {end_date} 的数据")
    
    # 按日期分块读取数据
    data_chunks = []
    
    for batch_idx in range(reader.num_record_batches):
        batch = reader.get_batch(batch_idx)
        chunk_df = batch.to_pandas()
        
        # 确保DateTime列存在
        if 'DateTime' not in chunk_df.columns:
            continue
            
        # 转换日期
        chunk_df['date'] = chunk_df['DateTime'].dt.date
        
        # 过滤日期范围
        mask = (chunk_df['date'] >= start_date) & (chunk_df['date'] <= end_date)
        chunk_df = chunk_df[mask]
        
        if len(chunk_df) == 0:
            continue
            
        data_chunks.append(chunk_df)
    
    if not data_chunks:
        print("警告：在指定日期范围内没有找到数据")
        return pd.DataFrame()
    
    df = pd.concat(data_chunks, ignore_index=True)
    print(f"数据加载完成，共 {len(df)} 行")
    return df

def train_xgboost_with_bayesian(file_path, 
                              target_col='10period_return', 
                              feature_cols=None, 
                              train_days=3,
                              val_days=1,
                              test_days=1,
                              min_train_samples=1000,
                              n_bayesian_iter=50,
                              output_dir="xgboost_bayesian_results",
                              standardize=True,
                              start_date=None,
                              end_date=None):
    """
    使用贝叶斯优化训练XGBoost模型，采用滚动窗口方式，流式加载数据
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化结果存储
    all_metrics = []
    window_results = []
    
    # 读取数据文件
    reader = pa.ipc.RecordBatchFileReader(pa.memory_map(file_path))
    
    # 如果没有指定日期范围，使用所有数据
    if start_date is None and end_date is None:
        print("使用所有数据")
        start_date = '2022-01-01'  # 设置一个默认的开始日期
        end_date = '2024-12-31'    # 设置一个默认的结束日期
    
    # 转换日期格式
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    print(f"加载 {start_date} 至 {end_date} 的数据")
    
    # 按批次读取数据
    current_window_data = []
    current_date = None
    
    for batch_idx in range(reader.num_record_batches):
        batch = reader.get_batch(batch_idx)
        chunk_df = batch.to_pandas()
        
        # 确保DateTime列存在
        if 'DateTime' not in chunk_df.columns:
            continue
            
        # 转换日期
        chunk_df['date'] = chunk_df['DateTime'].dt.date
        
        # 过滤日期范围
        mask = (chunk_df['date'] >= start_date) & (chunk_df['date'] <= end_date)
        chunk_df = chunk_df[mask]
        
        if len(chunk_df) == 0:
            continue
            
        # 按日期分组处理
        for date, group in chunk_df.groupby('date'):
            if date != current_date:
                # 处理前一个日期的数据
                if current_window_data:
                    window_df = pd.concat(current_window_data, ignore_index=True)
                    process_window(window_df, date, feature_cols, target_col, 
                                 train_days, val_days, test_days, min_train_samples,
                                 standardize, all_metrics, window_results)
                current_date = date
                current_window_data = []
            
            current_window_data.append(group)
    
    # 处理最后一个日期的数据
    if current_window_data:
        window_df = pd.concat(current_window_data, ignore_index=True)
        process_window(window_df, current_date, feature_cols, target_col, 
                      train_days, val_days, test_days, min_train_samples,
                      standardize, all_metrics, window_results)
    
    # 保存结果
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{output_dir}/all_window_metrics.csv", index=False)
        
        # 计算整体评估指标
        print("\n整体评估指标:")
        print(f"共训练了 {len(all_metrics)} 个窗口")
        
        window_ics = metrics_df['ic']
        print(f"窗口IC统计:")
        print(f"  平均: {window_ics.mean():.6f}")
        print(f"  最大: {window_ics.max():.6f}")
        print(f"  最小: {window_ics.min():.6f}")
        print(f"  标准差: {window_ics.std():.6f}")
        print(f"  正IC占比: {(window_ics > 0).mean() * 100:.2f}%")
        
        dir_acc = metrics_df['direction_accuracy']
        print(f"窗口方向准确率统计:")
        print(f"  平均: {dir_acc.mean():.6f}")
        print(f"  最大: {dir_acc.max():.6f}")
        print(f"  最小: {dir_acc.min():.6f}")
        print(f"  标准差: {dir_acc.std():.6f}")
        print(f"  优于随机(>0.5)占比: {(dir_acc > 0.5).mean() * 100:.2f}%")
        
        return {
            'window_results': window_results,
            'metrics': metrics_df
        }
    else:
        print("没有足够的数据进行训练")
        return None

def process_window(window_df, current_date, feature_cols, target_col, 
                  train_days, val_days, test_days, min_train_samples,
                  standardize, all_metrics, window_results):
    """处理单个窗口的数据"""
    # 数据预处理
    window_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if window_df.isnull().values.any():
        window_df.fillna(method='ffill', inplace=True)
    
    # 目标变量放大以提高数值精度
    scale_factor = 10000
    window_df[f'{target_col}_scaled'] = window_df[target_col] * scale_factor
    target_col_scaled = f'{target_col}_scaled'
    
    # 准备特征和标签
    date = pd.to_datetime(window_df['TradDay'])
    
    # 创建训练、验证和测试掩码
    train_mask = date == current_date
    val_mask = date == current_date + pd.Timedelta(days=1)
    test_mask = date == current_date + pd.Timedelta(days=2)
    
    # 准备当前窗口的数据
    X_train = window_df[train_mask][feature_cols]
    y_train = window_df[train_mask][target_col_scaled]
    X_val = window_df[val_mask][feature_cols]
    y_val = window_df[val_mask][target_col_scaled]
    X_test = window_df[test_mask][feature_cols]
    y_test = window_df[test_mask][target_col]
    
    print(f"\n处理日期 {current_date} 的数据:")
    print(f"训练样本: {X_train.shape[0]}")
    print(f"验证样本: {X_val.shape[0]}")
    print(f"测试样本: {X_test.shape[0]}")
    
    # 如果训练样本不足，跳过此窗口
    if len(X_train) < min_train_samples:
        print(f"警告: 训练样本不足 ({len(X_train)} < {min_train_samples})，跳过此窗口")
        return
    
    # 特征标准化
    if standardize:
        print("对特征进行标准化处理...")
        scaler = RobustScaler(quantile_range=(5, 95))
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # 处理极端值
        clip_threshold = 10.0
        for col in X_test_scaled.columns:
            X_test_scaled[col] = X_test_scaled[col].clip(-clip_threshold, clip_threshold)
            X_val_scaled[col] = X_val_scaled[col].clip(-clip_threshold, clip_threshold)
        
        X_train = X_train_scaled
        X_val = X_val_scaled
        X_test = X_test_scaled
    
    # 创建DMatrix数据
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dval = xgb.DMatrix(X_val.values, label=y_val.values)
    dtest = xgb.DMatrix(X_test.values, label=y_test.values)
    
    # 设置基础参数
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],
        'seed': 42
    }
    
    # 训练模型
    print("训练模型...")
    model = xgb.train(
        base_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # 在测试集上进行预测
    test_pred_scaled = model.predict(dtest)
    test_pred = test_pred_scaled / scale_factor
    
    # 计算评估指标
    if len(y_test) > 0 and not np.all(np.isnan(y_test)) and not np.all(np.isnan(test_pred)):
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_ic = calculate_ic(test_pred, y_test)
        
        # 计算方向准确率
        valid_mask = ~(np.isnan(y_test) | np.isnan(test_pred))
        if np.sum(valid_mask) > 0:
            test_direction_accuracy = np.mean(np.sign(y_test[valid_mask]) == np.sign(test_pred[valid_mask]))
        else:
            test_direction_accuracy = np.nan
        
        window_metric = {
            'date': current_date,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2,
            'ic': test_ic,
            'direction_accuracy': test_direction_accuracy
        }
        
        all_metrics.append(window_metric)
        
        # 保存窗口结果
        window_results.append({
            'date': current_date,
            'features_used': feature_cols,
            'metrics': window_metric,
            'feature_importance': model.get_score(importance_type='gain')
        })
        
        print(f"测试集评估指标:")
        print(f"MSE: {test_mse:.10f}, MAE: {test_mae:.10f}")
        print(f"R2: {test_r2:.6f}, IC: {test_ic:.6f}")
        print(f"方向准确率: {test_direction_accuracy:.6f}")
    
    # 释放内存
    del X_train, X_val, X_test, y_train, y_val, y_test, dtrain, dval, dtest
    del model
    import gc
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description='训练XGBoost模型，使用滚动窗口方式预测收益率')
    parser.add_argument('--train-days', type=int, default=3, help='训练窗口天数，默认3天')
    parser.add_argument('--val-days', type=int, default=1, help='验证窗口天数，默认1天')
    parser.add_argument('--test-days', type=int, default=1, help='测试窗口天数，默认1天')
    parser.add_argument('--n-iter', type=int, default=20, help='贝叶斯优化迭代次数，默认20次')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录，默认为自动生成')
    parser.add_argument('--feature-corr-cutoff', type=float, default=0.0, help='特征相关性阈值，用于初步筛选特征，默认0.0表示不筛选')
    parser.add_argument('--min-samples', type=int, default=1000, help='最小训练样本数，默认1000')
    parser.add_argument('--scale-factor', type=int, default=10000, help='目标变量缩放因子，默认10000')
    parser.add_argument('--standardize', action='store_true', help='是否对特征进行标准化处理，默认开启')
    parser.add_argument('--periods', type=int, default=10, help='计算period_return的周期，默认10')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期，格式：YYYY-MM-DD')
    args = parser.parse_args()
    
    if args.n_iter < 7:
        print(f"警告: 贝叶斯优化迭代次数 {args.n_iter} 小于最小要求(7)，已自动调整为10")
        args.n_iter = 10
    
    if args.output_dir is None:
        args.output_dir = f"xgboost_results_{args.train_days}_{args.val_days}_{args.test_days}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("注册所有因子...")
    factor_register.register_all_factors()
    
    print("获取因子列表...")
    manager = FactorManager()
    factors_list = manager.get_factor_names(frequency=FactorFrequency.TICK)
    
    print(f"找到 {len(factors_list)} 个因子特征")
    print("因子列表:", factors_list)
    
    # 获取第一个批次来确定可用的特征
    reader = pa.ipc.RecordBatchFileReader(pa.memory_map('data_with_factors.feather'))
    first_batch = reader.get_batch(0).to_pandas()
    available_factors = [factor for factor in factors_list if factor in first_batch.columns]
    print(f"数据集中可用的因子: {len(available_factors)}/{len(factors_list)}")
    
    if len(available_factors) == 0:
        print("错误: 数据集中没有可用的因子")
        return
    
    print("开始训练XGBoost模型...")
    start_time = datetime.now()
    results = train_xgboost_with_bayesian(
        file_path='data_with_factors.feather',
        target_col='10period_return',
        feature_cols=available_factors,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        min_train_samples=args.min_samples,
        n_bayesian_iter=args.n_iter,
        output_dir=args.output_dir,
        standardize=args.standardize,
        start_date=args.start_date,
        end_date=args.end_date
    )
    end_time = datetime.now()
    training_time = end_time - start_time
    
    if results is None:
        print("训练失败，可能是由于样本量不足")
        return
    
    print(f"\n训练总时间: {training_time}")
    print(f"平均每个窗口训练时间: {training_time / len(results['metrics'])}")
    
    print(f"\n结果已保存到 {args.output_dir} 目录")
    print("生成的汇总文件包括:")
    print(f"  - {args.output_dir}/all_window_metrics.csv       (每个窗口的基本评估指标)")
    
if __name__ == "__main__":
    main()
