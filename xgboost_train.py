import pandas as pd
import numpy as np
import xgboost_signal
import factor_register
from factor_manager import FactorManager, FactorFrequency
from factors_test import FactorsTester
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse

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
    parser.add_argument('--price-col', type=str, default='mid_price', help='用于计算收益率的价格列，默认mid_price')
    args = parser.parse_args()
    
    # 构建目标变量列名
    target_col = f'{args.periods}period_return'
    
    if args.n_iter < 7:
        print(f"警告: 贝叶斯优化迭代次数 {args.n_iter} 小于最小要求(7)，已自动调整为10")
        args.n_iter = 10
    
    if args.output_dir is None:
        args.output_dir = f"xgboost_results_{args.train_days}_{args.val_days}_{args.test_days}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("加载数据...")
    df = pd.read_feather('data_with_factors.feather')
    
    print("注册所有因子...")
    factor_register.register_all_factors()
    
    print("获取因子列表...")
    manager = FactorManager()
    factors_list = manager.get_factor_names(frequency=FactorFrequency.TICK)
    
    print(f"找到 {len(factors_list)} 个因子特征")
    print("因子列表:", factors_list)
    
    available_factors = [factor for factor in factors_list if factor in df.columns]
    print(f"数据集中可用的因子: {len(available_factors)}/{len(factors_list)}")
    
    if len(available_factors) == 0:
        print("错误: 数据集中没有可用的因子")
        return
    
    print(f"计算{args.periods}period_return...")
    # 使用FactorsTester.calculate_forward_returns方法计算收益率
    df = FactorsTester.calculate_forward_returns(
        df=df, 
        periods=[args.periods], 
        price_col=args.price_col
    )
    
    print(f"目标变量 '{target_col}' 统计:")
    print(f"  - 均值: {df[target_col].mean():.8f}")
    print(f"  - 标准差: {df[target_col].std():.8f}")
    print(f"  - 中位数: {df[target_col].median():.8f}")
    print(f"  - 最小值: {df[target_col].min():.8f}")
    print(f"  - 最大值: {df[target_col].max():.8f}")
    print(f"  - 非空值数量: {df[target_col].count()}")
    
    print("进行特征预处理...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    na_ratio = df[available_factors].isna().mean()
    print("特征缺失值比例:")
    for factor, ratio in na_ratio.items():
        print(f"  {factor}: {ratio:.4f}")
    
    good_factors = [f for f in available_factors if na_ratio[f] < 0.3]
    print(f"过滤后剩余特征: {len(good_factors)}/{len(available_factors)}")
    
    factor_std = df[good_factors].std()
    print("特征标准差:")
    for factor, std_val in factor_std.items():
        print(f"  {factor}: {std_val:.6f}")
    
    valid_factors = [f for f in good_factors if factor_std[f] > 1e-6]
    print(f"方差过滤后剩余特征: {len(valid_factors)}/{len(good_factors)}")
    
    if len(valid_factors) == 0:
        print("错误: 没有有效的特征可用于训练")
        return
    
    if args.feature_corr_cutoff > 0:
        print(f"基于特征与目标变量的相关性筛选特征 (阈值: {args.feature_corr_cutoff})...")
        correlations = {}
        for feature in valid_factors:
            corr = np.abs(df[[feature, target_col]].corr().iloc[0, 1])
            correlations[feature] = corr
        
        selected_features = [f for f, c in correlations.items() if c > args.feature_corr_cutoff]
        print(f"相关性筛选后剩余特征: {len(selected_features)}/{len(valid_factors)}")
        
        if len(selected_features) < 10:
            print("警告: 相关性筛选后特征太少，使用所有有效特征")
            selected_features = valid_factors
    else:
        selected_features = valid_factors
    
    print("删除含有缺失值的行...")
    rows_before = len(df)
    df.dropna(subset=[target_col] + selected_features, inplace=True)
    rows_after = len(df)
    print(f"删除NaN后剩余样本数: {rows_after} (删除了 {rows_before - rows_after} 行)")
    
    print(f"开始训练XGBoost模型，使用滚动窗口: 训练{args.train_days}天, 验证{args.val_days}天, 测试{args.test_days}天...")
    print(f"贝叶斯优化迭代次数: {args.n_iter}")
    print(f"目标变量缩放因子: {args.scale_factor}")
    print(f"最小训练样本数: {args.min_samples}")
    
    start_time = datetime.now()
    results = xgboost_signal.train_xgboost_with_bayesian(
        df=df,
        target_col=target_col,  # 使用动态生成的目标列名
        feature_cols=selected_features,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        min_train_samples=args.min_samples,
        n_bayesian_iter=args.n_iter,
        output_dir=args.output_dir,
        standardize=args.standardize
    )
    end_time = datetime.now()
    training_time = end_time - start_time
    
    if results is None:
        print("训练失败，可能是由于样本量不足")
        return
    
    print("\n整体模型评估结果:")
    print(f"整体 MSE: {results['overall_metrics']['overall_mse']:.15f}")
    print(f"整体 R2: {results['overall_metrics']['overall_r2']:.6f}")
    print(f"整体 IC: {results['overall_metrics']['overall_ic']:.6f}")
    print(f"整体方向准确率: {results['overall_metrics']['direction_accuracy']:.6f}")
    print(f"策略夏普比率: {results['overall_metrics'].get('sharpe_ratio', float('nan')):.6f}")
    print(f"策略信息比率: {results['overall_metrics'].get('information_ratio', float('nan')):.6f}")
    
    print(f"\n共训练了 {len(results['metrics'])} 个窗口")
    
    window_ics = results['metrics']['ic']
    print(f"窗口IC统计:")
    print(f"  平均: {window_ics.mean():.6f}")
    print(f"  最大: {window_ics.max():.6f}")
    print(f"  最小: {window_ics.min():.6f}")
    print(f"  标准差: {window_ics.std():.6f}")
    print(f"  正IC占比: {(window_ics > 0).mean() * 100:.2f}%")
    
    dir_acc = results['metrics']['direction_accuracy']
    print(f"窗口方向准确率统计:")
    print(f"  平均: {dir_acc.mean():.6f}")
    print(f"  最大: {dir_acc.max():.6f}")
    print(f"  最小: {dir_acc.min():.6f}")
    print(f"  标准差: {dir_acc.std():.6f}")
    print(f"  优于随机(>0.5)占比: {(dir_acc > 0.5).mean() * 100:.2f}%")
    
    print(f"\n训练总时间: {training_time}")
    print(f"平均每个窗口训练时间: {training_time / len(results['metrics'])}")
    
    print(f"\n结果已保存到 {args.output_dir} 目录")
    print("生成的汇总文件包括:")
    print(f"  - {args.output_dir}/window_metrics_summary.csv   (所有窗口评估指标的统计摘要)")
    print(f"  - {args.output_dir}/overall_metrics_summary.csv  (整体评估指标汇总)")
    print(f"  - {args.output_dir}/detailed_window_metrics.csv  (每个窗口的详细指标和参数)")
    print(f"  - {args.output_dir}/all_window_metrics.csv       (每个窗口的基本评估指标)")
    print(f"  - {args.output_dir}/avg_feature_importance.csv   (特征重要性排名)")
    print(f"  - {args.output_dir}/all_predictions.csv          (所有预测结果)")
    print(f"  - {args.output_dir}/param_performance_correlation.csv (参数与性能指标的相关性分析)")
    print(f"  - {args.output_dir}/performance_group_analysis.csv (按IC值分组的窗口分析)")
    
    print("\n可视化结果包括:")
    print(f"  - {args.output_dir}/avg_feature_importance.png   (Feature Importance Visualization)")
    print(f"  - {args.output_dir}/window_ic_and_direction.png  (IC and Direction Accuracy by Window)")
    print(f"  - {args.output_dir}/window_r2_and_mse.png        (R² and MSE Trends by Window)")
    print(f"  - {args.output_dir}/predictions_vs_returns.png   (Predicted vs Actual Returns)")
    print(f"  - {args.output_dir}/quantile_returns.png         (Return Analysis by Prediction Quantile)")
    print(f"  - {args.output_dir}/param_performance_heatmap.png (Parameter-Performance Correlation Heatmap)")
    
    if not results['feature_importance'].empty and len(results['feature_importance']) > 0:
        top_features = results['feature_importance'].head(10)
        print("\n重要特征 Top 10:")
        for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance_mean'])):
            usage = top_features.iloc[i]['usage_pct']
            print(f"{i+1}. {feature}: 重要性={importance:.6f}, 使用率={usage:.2f}%")

if __name__ == "__main__":
    main()
