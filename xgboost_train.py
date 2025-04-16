import pandas as pd
import numpy as np
import xgboost as xgb
import xgboost_signal
import factor_register
from factor_manager import FactorManager, FactorFrequency
from factors_test import FactorsTester
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse
from scipy import stats
import skopt
from skopt.space import Real, Integer, Categorical
import warnings

# 忽略XGBoost警告
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

def main():
    parser = argparse.ArgumentParser(description='训练XGBoost回归模型，使用滚动窗口方式预测收益率')
    parser.add_argument('--train-days', type=int, default=3, help='训练窗口天数，默认3天')
    parser.add_argument('--val-days', type=int, default=1, help='验证窗口天数，默认1天')
    parser.add_argument('--test-days', type=int, default=1, help='测试窗口天数，默认1天')
    parser.add_argument('--n-iter', type=int, default=20, help='贝叶斯优化迭代次数，默认20次')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录，默认为自动生成')
    parser.add_argument('--feature-corr-cutoff', type=float, default=0.0, help='特征相关性阈值，用于初步筛选特征，默认0.0表示不筛选')
    parser.add_argument('--min-samples', type=int, default=1000, help='最小训练样本数，默认1000')
    parser.add_argument('--standardize', action='store_true', help='是否对特征进行标准化处理，默认开启')
    parser.add_argument('--periods', type=int, default=10, help='计算period_return的周期，默认10')
    parser.add_argument('--price-col', type=str, default='mid_price', help='用于计算收益率的价格列，默认mid_price')
    parser.add_argument('--signal-threshold', type=float, default=0, help='信号强度阈值，预测收益率绝对值大于此值才视为有效信号，默认0.0005')
    parser.add_argument('--prob-mapping', type=str, choices=['sigmoid', 'abs', 'custom'], default='sigmoid', 
                        help='预测值到概率的映射方法：sigmoid(S型函数)、abs(绝对值归一化)或custom(自定义缩放)')
    parser.add_argument('--prob-threshold', type=float, default=0.53, help='概率过滤阈值，概率大于此值才视为有效信号，默认0.53')
    parser.add_argument('--prob-scale', type=float, default=1, help='概率映射的缩放因子，用于sigmoid和custom方法，默认10.0')
    parser.add_argument('--abs-scale', type=float, default=0.001, help='abs方法的固定缩放因子，将预测值除以此值后应用sigmoid，默认0.001')
    parser.add_argument('--loss-function', type=str, choices=['mse', 'direction', 'custom'], default='mse',
                       help='损失函数类型: mse(均方误差)、direction(方向损失)、custom(自定义损失)')
    args = parser.parse_args()
    
    # 构建目标变量列名
    target_col = f'{args.periods}period_return'
    
    if args.n_iter < 7:
        print(f"警告: 贝叶斯优化迭代次数 {args.n_iter} 小于最小要求(7)，已自动调整为10")
        args.n_iter = 10
    
    if args.output_dir is None:
        args.output_dir = f"xgboost_reg_results_{args.train_days}_{args.val_days}_{args.test_days}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("加载数据...")
    df = pd.read_feather('data_with_factors_and_smoothed.feather')
    
    print("注册所有因子...")
    factor_register.register_all_factors()
    
    print("获取因子列表...")
    manager = FactorManager()
    factors_list = ['momentum_10_ema_1200', 'weighted_momentum_10_ema_1200', 'momentum_20_ema_1200', 'weighted_momentum_20_ema_1200', 'momentum_50_ema_1200', 'weighted_momentum_50_ema_1200', 'momentum_100_ema_1200', 'weighted_momentum_100_ema_1200', 'realized_vol_50_ema_1200', 'high_low_vol_50_ema_1200', 'realized_vol_100_ema_1200', 'high_low_vol_100_ema_1200', 'realized_vol_200_ema_1200', 'high_low_vol_200_ema_1200', 'volume_intensity_100_ema_1200', 'order_book_imbalance_ema_1200', 'effective_spread_ema_1200', 'amihud_illiquidity_ema_1200', 'order_flow_toxicity_ema_1200', 'volume_synchronized_probability_ema_1200', 'bid_ask_pressure_ema_1200', 'price_impact_ema_1200', 'quote_slope_ema_1200', 'price_reversal_ema_1200', 'hft_trend_ema_1200', 'microstructure_momentum_ema_1200', 'intraday_seasonality_ema_1200', 'term_premium_ema_1200', 'volume_price_trend_ema_1200', 'liquidity_adjusted_momentum_ema_1200', 'momentum_10', 'weighted_momentum_10', 'momentum_20', 'weighted_momentum_20', 'momentum_50', 'weighted_momentum_50', 'momentum_100', 'weighted_momentum_100', 'realized_vol_50', 'high_low_vol_50', 'realized_vol_100', 'high_low_vol_100', 'realized_vol_200', 'high_low_vol_200', 'volume_intensity_100', 'order_book_imbalance', 'effective_spread', 'amihud_illiquidity', 'order_flow_toxicity', 'volume_synchronized_probability', 'bid_ask_pressure', 'price_impact', 'quote_slope', 'price_reversal', 'hft_trend', 'microstructure_momentum', 'intraday_seasonality', 'term_premium', 'volume_price_trend', 'liquidity_adjusted_momentum']
    
    print(f"找到 {len(factors_list)} 个因子特征")
    print("因子列表:", factors_list)
    
    available_factors = [factor for factor in factors_list if factor in df.columns]
    print(f"数据集中可用的因子: {len(available_factors)}/{len(factors_list)}")
    
    if len(available_factors) == 0:
        print("错误: 数据集中没有可用的因子")
        return
    
    print(f"计算{args.periods}period_return...")
    df = FactorsTester.calculate_forward_returns(
        df=df, 
        periods=[args.periods], 
        price_col=args.price_col,
        max_value=0.05  # 限制收益率最大为5%
    )
    
    # 处理异常值
    print("处理收益率异常值...")
    df[target_col].replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[~df[target_col].isna()]  # 删除收益率为NaN的行
    
    # 输出收益率统计
    print(f"清洗后目标变量 '{target_col}' 统计:")
    print(f"  - 均值: {df[target_col].mean():.8f}")
    print(f"  - 标准差: {df[target_col].std():.8f}")
    print(f"  - 中位数: {df[target_col].median():.8f}")
    print(f"  - 最小值: {df[target_col].min():.8f}")
    print(f"  - 最大值: {df[target_col].max():.8f}")
    print(f"  - 非空值数量: {df[target_col].count()}")
    
    # 计算实际方向
    direction_col = f'{target_col}_direction'
    df[direction_col] = np.sign(df[target_col])
    
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
    
    print(f"开始训练XGBoost回归模型，使用滚动窗口: 训练{args.train_days}天, 验证{args.val_days}天, 测试{args.test_days}天...")
    print(f"贝叶斯优化迭代次数: {args.n_iter}")
    print(f"最小训练样本数: {args.min_samples}")
    print(f"信号强度阈值: {args.signal_threshold}")
    
    start_time = datetime.now()
    
    # 定义存储结果的字典
    results = train_xgboost_regression(
        df=df,
        target_col=target_col,  # 使用收益率列作为目标
        feature_cols=selected_features,
        direction_col=direction_col,  # 使用方向列进行准确率计算
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        min_train_samples=args.min_samples,
        n_bayesian_iter=args.n_iter,
        output_dir=args.output_dir,
        standardize=args.standardize,
        signal_threshold=args.signal_threshold,
        prob_mapping=args.prob_mapping,
        prob_threshold=args.prob_threshold,
        prob_scale=args.prob_scale,
        abs_scale=args.abs_scale,
        loss_function=args.loss_function
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    if results is None:
        print("训练失败，可能是由于样本量不足")
        return
    
    print("\n整体模型评估结果:")
    print(f"整体R2分数: {results['overall_metrics']['r2_score']:.6f}")
    print(f"整体有效信号比例: {results['overall_metrics']['valid_signal_ratio']:.6f}")
    
    if 'valid_direction_accuracy' in results['overall_metrics'] and not np.isnan(results['overall_metrics']['valid_direction_accuracy']):
        print(f"有效信号方向准确率: {results['overall_metrics']['valid_direction_accuracy']:.6f}")
    
    if 'valid_ic' in results['overall_metrics'] and not np.isnan(results['overall_metrics']['valid_ic']):
        print(f"有效信号IC: {results['overall_metrics']['valid_ic']:.6f}")
    
    if 'all_ic' in results['overall_metrics'] and not np.isnan(results['overall_metrics']['all_ic']):
        print(f"所有信号IC: {results['overall_metrics']['all_ic']:.6f}")
    
    print(f"整体信号效率: {results['overall_metrics']['signal_efficiency']:.6f}")
    
    # 显示信号类型分布情况
    if 'signal_type' in results['all_predictions']:
        signal_types = results['all_predictions']['signal_type']
        up_signals = sum(1 for s in signal_types if s == 1)
        down_signals = sum(1 for s in signal_types if s == -1)
        no_signals = sum(1 for s in signal_types if s == 0)
        total_signals = len(signal_types)
        
        print("\n信号类型分布:")
        print(f"  - 上涨信号: {up_signals} ({up_signals/total_signals*100:.2f}%)")
        print(f"  - 下跌信号: {down_signals} ({down_signals/total_signals*100:.2f}%)")
        print(f"  - 无信号/弱信号: {no_signals} ({no_signals/total_signals*100:.2f}%)")
    
    print(f"\n共训练了 {len(results['metrics'])} 个窗口")
    
    if not results['feature_importance'].empty and len(results['feature_importance']) > 0:
        # 获取特征重要性数据
        feature_importance = results['feature_importance'].copy()
        
        # 1. 归一化特征重要性值 (使总和为1)
        if 'importance_mean' in feature_importance.columns:
            total_importance = feature_importance['importance_mean'].sum()
            if total_importance > 0:
                feature_importance['importance_norm'] = feature_importance['importance_mean'] / total_importance * 100
            else:
                feature_importance['importance_norm'] = 0.0
        
        # 2. 修正使用率计算，确保不超过100%
        if 'usage_pct' in feature_importance.columns:
            feature_importance['usage_pct'] = feature_importance['usage_pct'].clip(0, 100)
        
        # 3. 获取前10个特征
        top_features = feature_importance.head(10)
        
        print("\n重要特征 Top 10:")
        for i, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance_norm'] if 'importance_norm' in row else row['importance_mean']
            usage = row['usage_pct'] if 'usage_pct' in row else 0.0
            print(f"{i+1}. {feature}: 重要性={importance:.2f}%, 使用率={usage:.2f}%")
        
        # 保存归一化后的特征重要性
        feature_importance.to_csv(f"{args.output_dir}/normalized_feature_importance.csv", index=False)
    
    print(f"\n训练总时间: {training_time}")
    print(f"平均每个窗口训练时间: {training_time / len(results['metrics'])}")
    
    print(f"\n结果已保存到 {args.output_dir} 目录")
    print("生成的汇总文件包括:")
    print(f"  - {args.output_dir}/window_metrics_summary.csv   (所有窗口评估指标的统计摘要)")
    print(f"  - {args.output_dir}/overall_metrics_summary.csv  (整体评估指标汇总)")
    print(f"  - {args.output_dir}/predictions.csv        (包含原始预测、过滤后预测和是否有效信号的完整结果)")
    print(f"  - {args.output_dir}/normalized_feature_importance.csv (归一化特征重要性)")

def train_xgboost_regression(df, target_col, feature_cols, direction_col, 
                            train_days, val_days, test_days, min_train_samples,
                            n_bayesian_iter, output_dir, standardize, signal_threshold,
                            prob_mapping, prob_threshold, prob_scale, abs_scale=0.001,
                            loss_function='mse'):
    """
    使用贝叶斯优化训练XGBoost回归模型，采用滚动窗口方式
    
    Args:
        df: 包含特征和标签的DataFrame
        target_col: 目标列名 (收益率)
        feature_cols: 特征列列表
        direction_col: 方向列名 (1表示上涨，-1表示下跌)
        train_days: 训练窗口天数
        val_days: 验证窗口天数
        test_days: 测试窗口天数
        min_train_samples: 最小训练样本数
        n_bayesian_iter: 贝叶斯优化迭代次数
        output_dir: 输出目录
        standardize: 是否对特征进行标准化处理
        signal_threshold: 信号强度阈值，预测收益率绝对值大于此值才视为有效信号
        prob_mapping: 概率映射方法
        prob_threshold: 概率过滤阈值
        prob_scale: 概率映射的缩放因子
        abs_scale: abs方法的固定缩放因子
        loss_function: 损失函数类型
        
    Returns:
        包含模型评估指标和预测结果的字典
    """
    # 设置输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有唯一的交易日
    all_days = sorted(df['TradDay'].unique())
    if len(all_days) < train_days + val_days + test_days:
        print(f"错误: 数据中的交易日 ({len(all_days)}) 少于所需的训练+验证+测试天数 ({train_days}+{val_days}+{test_days})")
        return None
    
    # 用于存储所有窗口的预测结果
    all_predictions = []
    all_actual_returns = []
    all_probabilities = []
    all_dates = []
    all_instruments = []
    all_valid_signals = []
    all_signal_types = []
    
    # 用于存储每个窗口的评估指标
    metrics_list = []
    
    # 存储特征重要性数据
    feature_importance_list = []
    
    # 定义映射预测到概率的函数
    def map_prediction_to_probability(pred, method='sigmoid', scale=10.0, abs_scale=0.001):
        if method == 'sigmoid':
            # 使用sigmoid函数映射到(0,1)区间
            return 1 / (1 + np.exp(-pred * scale))
        elif method == 'custom':
            # 客制化概率映射，可以根据预测值分布调整
            # 这里实现一个简单的分段函数
            probabilities = np.zeros_like(pred)
            pos_mask = pred > 0
            neg_mask = pred < 0
            
            # 正向预测映射到(0.5,1)
            probabilities[pos_mask] = 0.5 + 0.5 * np.tanh(pred[pos_mask] * scale)
            # 负向预测映射到(0,0.5)
            probabilities[neg_mask] = 0.5 - 0.5 * np.tanh(-pred[neg_mask] * scale)
            # 预测为0的部分设为0.5
            probabilities[pred == 0] = 0.5
            
            return probabilities
        elif method == 'abs':
            # 修复信息泄漏问题：使用固定缩放因子而不是基于测试集的最大值
            # 使用sigmoid确保平滑映射到(0,1)区间
            abs_pred = np.abs(pred)
            # 使用固定缩放因子
            scaled_pred = abs_pred / abs_scale
            # 应用sigmoid确保值域在(0,1)
            return 1 / (1 + np.exp(-scaled_pred))
        else:
            # 默认方法，简单线性映射后截断
            return np.clip(0.5 + pred * scale, 0, 1)
    
    # 自定义方向损失函数，强调方向预测的准确性
    def direction_objective(y_pred, dtrain):
        y_true = dtrain.get_label()
        # 计算方向一致性
        direction_match = np.sign(y_pred) * np.sign(y_true)
        # 计算预测误差
        squared_error = (y_pred - y_true) ** 2
        # 组合损失：方向不匹配时加大惩罚
        combined_loss = squared_error * (1.0 + np.where(direction_match < 0, 1.0, 0.0))
        grad = 2 * (y_pred - y_true) * (1.0 + np.where(direction_match < 0, 1.0, 0.0))
        hess = 2.0 * np.ones_like(y_pred)
        return grad, hess
    
    # 自定义评估指标，评估方向准确率
    def direction_metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        direction_correct = np.sum(np.sign(y_pred) == np.sign(y_true))
        direction_accuracy = direction_correct / len(y_true)
        return 'direction_acc', direction_accuracy
    
    # 遍历训练窗口
    for i in range(0, len(all_days) - (train_days + val_days + test_days) + 1):
        window_start = all_days[i]
        train_end = all_days[i + train_days - 1]
        val_end = all_days[i + train_days + val_days - 1]
        test_end = all_days[i + train_days + val_days + test_days - 1]
        
        print(f"\n[窗口 {i+1}/{len(all_days)-(train_days+val_days+test_days)+1}]")
        print(f"训练: {window_start} 至 {train_end}")
        print(f"验证: {train_end} 至 {val_end}")
        print(f"测试: {val_end} 至 {test_end}")
        
        # 分割数据
        train_mask = (df['TradDay'] >= window_start) & (df['TradDay'] <= train_end)
        val_mask = (df['TradDay'] > train_end) & (df['TradDay'] <= val_end)
        test_mask = (df['TradDay'] > val_end) & (df['TradDay'] <= test_end)
        
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, target_col]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, target_col]
        
        # 对目标变量（收益率）进行标准化
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        # 只使用训练集数据拟合scaler，避免信息泄露
        y_train_scaled = pd.Series(y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
        # 使用相同参数转换验证集和测试集
        y_val_scaled = pd.Series(y_scaler.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index)
        y_test_scaled = pd.Series(y_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
        
        # 收集原始收益率，用于评估和IC计算
        returns_train = df.loc[train_mask, target_col]
        returns_val = df.loc[val_mask, target_col]
        returns_test = df.loc[test_mask, target_col]
        
        # 收集日期和合约信息，用于结果分析
        dates_test = df.loc[test_mask, 'TradDay']
        instruments_test = df.loc[test_mask, 'InstruID']
        
        # 检查训练样本数量
        if len(X_train) < min_train_samples:
            print(f"训练样本数 {len(X_train)} 小于最小要求 {min_train_samples}，跳过此窗口")
            continue
        
        # 特征标准化
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # 定义超参数空间
        space = [
            Integer(3, 15, name='max_depth'),
            Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
            Real(0.1, 10, prior='log-uniform', name='gamma'),
            Real(1, 10, prior='uniform', name='min_child_weight'),
            Real(0.1, 1.0, prior='uniform', name='subsample'),
            Real(0.1, 1.0, prior='uniform', name='colsample_bytree'),
            Real(0, 10, prior='uniform', name='reg_alpha'),
            Real(0, 10, prior='uniform', name='reg_lambda'),
        ]
        
        # 选择损失函数
        if loss_function == 'mse':
            space.append(Categorical(['reg:squarederror'], name='objective'))
        
        # 定义目标函数
        @skopt.utils.use_named_args(space)
        def objective_function(**params):
            # 准备评估用的数据集
            dtrain = xgb.DMatrix(X_train, label=y_train_scaled)  # 使用标准化后的目标变量
            dval = xgb.DMatrix(X_val, label=y_val_scaled)  # 使用标准化后的目标变量
            
            # 设置评估指标
            if loss_function == 'direction':
                # 对于direction损失函数，我们删除params中的objective参数(如果存在)
                if 'objective' in params:
                    del params['objective']  # 移除内置objective，使用自定义的
                
                evals_result = {}
                # 训练模型
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False,  # 不显示训练进度
                    evals_result=evals_result,
                    obj=direction_objective,  # 直接使用函数对象
                    feval=direction_metric    # 直接使用函数对象
                )
            else:
                # MSE或其他损失函数
                evals_result = {}
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False,  # 不显示训练进度
                    evals_result=evals_result
                )
            
            # 获取特征重要性
            importance = model.get_score(importance_type='gain')
            for feature, score in importance.items():
                feature_importance_list.append({
                    'window': i,
                    'feature': feature,
                    'importance': score,
                    'used': 1
                })
            
            # 预测测试集（获得标准化尺度的预测）
            y_test_pred_scaled = model.predict(xgb.DMatrix(X_test))
            
            # 将预测值转换回原始尺度
            y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
            
            # 输出预测值的基本统计信息，帮助分析
            pred_abs_mean = np.abs(y_test_pred).mean()
            pred_max = np.max(np.abs(y_test_pred))
            pred_min = np.min(np.abs(y_test_pred))
            
            # 将预测值映射到概率空间
            probabilities = map_prediction_to_probability(
                y_test_pred, 
                method=prob_mapping,
                scale=prob_scale,
                abs_scale=abs_scale
            )
            
            # 使用概率阈值进行过滤
            if prob_mapping == 'abs':
                # abs方法需要特殊处理，因为它是将所有值映射到[0,1]，没有上下方向之分
                # 我们可以根据原始预测值的正负来确定上涨下跌，并用abs方法的概率来确定置信度
                up_signals = (y_test_pred > 0) & (probabilities >= prob_threshold)
                down_signals = (y_test_pred < 0) & (probabilities >= prob_threshold)
                valid_signal_test = up_signals | down_signals
                
                # 计算信号类型
                signal_types = np.zeros_like(y_test_pred)
                signal_types[up_signals] = 1
                signal_types[down_signals] = -1
            else:
                # 实现三态信号过滤：
                # 1. 概率大于阈值 -> 保留为上涨信号
                up_signals = probabilities >= prob_threshold
                # 2. 概率小于(1-阈值) -> 保留为下跌信号
                down_signals = probabilities <= (1 - prob_threshold)
                # 3. 其他 -> 过滤掉
                valid_signal_test = up_signals | down_signals
                
                # 计算信号类型
                signal_types = np.zeros_like(y_test_pred)
                signal_types[up_signals] = 1
                signal_types[down_signals] = -1
                
                # 强度过滤可以作为辅助条件
                if signal_threshold > 0:
                    strong_signal = np.abs(y_test_pred) >= signal_threshold
                    valid_signal_test = valid_signal_test & strong_signal
                    # 如果不满足强度要求，也将信号类型设为0
                    signal_types[~strong_signal] = 0
            
            # 计算指标
            mse = mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            
            # 只对有效信号计算方向准确率和IC
            if valid_signal_test.sum() > 0:
                # 计算有效信号的方向准确率
                pred_direction = np.sign(y_test_pred[valid_signal_test])
                true_direction = np.sign(y_test.values[valid_signal_test])
                direction_accuracy = accuracy_score(true_direction, pred_direction)
                
                # 计算有效信号的IC值
                valid_ic = np.corrcoef(y_test_pred[valid_signal_test], returns_test.values[valid_signal_test])[0, 1]
                
                # 计算全部信号的IC值
                ic = np.corrcoef(y_test_pred, returns_test.values)[0, 1]
            else:
                direction_accuracy = np.nan
                valid_ic = np.nan
                ic = np.corrcoef(y_test_pred, returns_test.values)[0, 1] if len(y_test_pred) > 1 else np.nan
            
            # 存储指标和预测结果
            metrics = {
                'window': i,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'mse': mse,
                'r2_score': r2,
                'direction_accuracy': direction_accuracy,
                'ic': ic,
                'valid_ic': valid_ic,
                'valid_signal_ratio': valid_signal_test.mean(),
                **params
            }
            metrics_list.append(metrics)
            
            # 存储所有窗口的预测结果
            all_predictions.extend(y_test_pred.tolist())
            all_actual_returns.extend(returns_test.values.tolist())
            all_dates.extend(dates_test.values.tolist())
            all_instruments.extend(instruments_test.values.tolist())
            all_valid_signals.extend(valid_signal_test.tolist())
            all_probabilities.extend(probabilities.tolist())
            all_signal_types.extend(signal_types.tolist())
            
            # 优化过程中减少输出
            # print(f"窗口 {i} 训练完成，测试集MSE: {mse:.4f}")
            # print(f"测试集R2分数: {r2:.4f}")
            # print(f"有效信号比例: {valid_signal_test.mean():.4f}")
            # if not np.isnan(direction_accuracy):
            #     print(f"有效信号方向准确率: {direction_accuracy:.4f}")
            # if not np.isnan(valid_ic):
            #     print(f"有效信号IC: {valid_ic:.4f}")
                
            # 根据损失函数返回不同的优化目标
            if loss_function == 'direction':
                # 方向损失函数优化方向准确率，如果无有效信号则优化R2
                if not np.isnan(direction_accuracy) and valid_signal_test.sum() > 0:
                    return -direction_accuracy  # 负号表示最小化负准确率 = 最大化准确率
                else:
                    return -r2  # 如果没有有效信号，优化R2分数
            else:
                # MSE损失函数优化R2分数
                return -r2  # 负号表示最小化负R2 = 最大化R2
        
        # 执行贝叶斯优化
        print(f"开始贝叶斯优化，迭代次数: {n_bayesian_iter}")
        print(f"使用损失函数: {loss_function}")
        # 使用tqdm显示优化进度
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = skopt.gp_minimize(objective_function, space, n_calls=n_bayesian_iter, random_state=42)
        
        # 最佳参数
        best_params = {
            'max_depth': result.x[0],
            'learning_rate': result.x[1],
            'gamma': result.x[2],
            'min_child_weight': result.x[3],
            'subsample': result.x[4],
            'colsample_bytree': result.x[5],
            'reg_alpha': result.x[6],
            'reg_lambda': result.x[7],
            'seed': 42
        }
        
        # 如果使用MSE，添加objective参数
        if loss_function == 'mse':
            best_params['objective'] = 'reg:squarederror'
            best_params['eval_metric'] = 'rmse'  # 使用内置的rmse评估指标
        # 注意：direction损失函数不在此处设置eval_metric，而是在train函数中通过feval参数传递
        
        print(f"\n窗口 {i+1} - 贝叶斯优化完成")
        print("最佳参数:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # 使用最佳参数训练最终模型
        dtrain = xgb.DMatrix(X_train, label=y_train_scaled)  # 使用标准化后的目标变量
        dval = xgb.DMatrix(X_val, label=y_val_scaled)  # 使用标准化后的目标变量
        evals_result = {}
        
        # 如果使用方向损失函数，添加自定义objective和eval
        if loss_function == 'direction':
            model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False,  # 不输出中间信息
                evals_result=evals_result,
                obj=direction_objective,
                feval=direction_metric
            )
        else:
            # 如果使用标准MSE，添加objective
            if loss_function == 'mse':
                best_params['objective'] = 'reg:squarederror'
            
            model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False,  # 不输出中间信息
                evals_result=evals_result
            )
        
        # 获取特征重要性
        importance = model.get_score(importance_type='gain')
        for feature, score in importance.items():
            feature_importance_list.append({
                'window': i,
                'feature': feature,
                'importance': score,
                'used': 1
            })
        
        # 预测测试集（获得标准化尺度的预测）
        y_test_pred_scaled = model.predict(xgb.DMatrix(X_test))
        
        # 将预测值转换回原始尺度
        y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # 输出预测值的基本统计信息，帮助分析
        pred_abs_mean = np.abs(y_test_pred).mean()
        pred_max = np.max(np.abs(y_test_pred))
        pred_min = np.min(np.abs(y_test_pred))
        
        # 将预测值映射到概率空间
        probabilities = map_prediction_to_probability(
            y_test_pred, 
            method=prob_mapping,
            scale=prob_scale,
            abs_scale=abs_scale
        )
        
        # 使用概率阈值进行过滤
        if prob_mapping == 'abs':
            # abs方法需要特殊处理，因为它是将所有值映射到[0,1]，没有上下方向之分
            # 我们可以根据原始预测值的正负来确定上涨下跌，并用abs方法的概率来确定置信度
            up_signals = (y_test_pred > 0) & (probabilities >= prob_threshold)
            down_signals = (y_test_pred < 0) & (probabilities >= prob_threshold)
            valid_signal_test = up_signals | down_signals
            
            # 计算信号类型
            signal_types = np.zeros_like(y_test_pred)
            signal_types[up_signals] = 1
            signal_types[down_signals] = -1
        else:
            # 实现三态信号过滤：
            # 1. 概率大于阈值 -> 保留为上涨信号
            up_signals = probabilities >= prob_threshold
            # 2. 概率小于(1-阈值) -> 保留为下跌信号
            down_signals = probabilities <= (1 - prob_threshold)
            # 3. 其他 -> 过滤掉
            valid_signal_test = up_signals | down_signals
            
            # 计算信号类型
            signal_types = np.zeros_like(y_test_pred)
            signal_types[up_signals] = 1
            signal_types[down_signals] = -1
            
            # 强度过滤可以作为辅助条件
            if signal_threshold > 0:
                strong_signal = np.abs(y_test_pred) >= signal_threshold
                valid_signal_test = valid_signal_test & strong_signal
                # 如果不满足强度要求，也将信号类型设为0
                signal_types[~strong_signal] = 0
        
        # 输出信号统计
        up_count = np.sum(signal_types==1)
        down_count = np.sum(signal_types==-1)
        no_signal_count = np.sum(signal_types==0)
        print(f"信号统计: 上涨={up_count}, 下跌={down_count}, 无信号={no_signal_count}")
        print(f"有效信号比例: {valid_signal_test.mean():.4f}")
        
        # 计算指标
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # 只对有效信号计算方向准确率和IC
        if valid_signal_test.sum() > 0:
            # 计算有效信号的方向准确率
            pred_direction = np.sign(y_test_pred[valid_signal_test])
            true_direction = np.sign(y_test.values[valid_signal_test])
            direction_accuracy = accuracy_score(true_direction, pred_direction)
            
            # 计算有效信号的IC值
            valid_ic = np.corrcoef(y_test_pred[valid_signal_test], returns_test.values[valid_signal_test])[0, 1]
            
            # 计算全部信号的IC值
            ic = np.corrcoef(y_test_pred, returns_test.values)[0, 1]
        else:
            direction_accuracy = np.nan
            valid_ic = np.nan
            ic = np.corrcoef(y_test_pred, returns_test.values)[0, 1] if len(y_test_pred) > 1 else np.nan
        
        # 存储指标和预测结果
        metrics = {
            'window': i,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'mse': mse,
            'r2_score': r2,
            'direction_accuracy': direction_accuracy,
            'ic': ic,
            'valid_ic': valid_ic,
            'valid_signal_ratio': valid_signal_test.mean(),
            **best_params
        }
        metrics_list.append(metrics)
        
        # 存储所有窗口的预测结果
        all_predictions.extend(y_test_pred.tolist())
        all_actual_returns.extend(returns_test.values.tolist())
        all_dates.extend(dates_test.values.tolist())
        all_instruments.extend(instruments_test.values.tolist())
        all_valid_signals.extend(valid_signal_test.tolist())
        all_probabilities.extend(probabilities.tolist())
        all_signal_types.extend(signal_types.tolist())
        
        # 输出最终结果
        print(f"窗口 {i+1} 最终模型结果:")
        print(f"  测试集MSE: {mse:.6f}")
        print(f"  测试集R2分数: {r2:.6f}")
        if not np.isnan(direction_accuracy):
            print(f"  有效信号方向准确率: {direction_accuracy:.6f}")
        if not np.isnan(valid_ic):
            print(f"  有效信号IC: {valid_ic:.6f}")
        print("---------------------------------------------------")
    
    # 如果没有成功训练任何窗口，返回None
    if not metrics_list:
        print("没有成功训练任何窗口")
        return None
    
    # 整合所有窗口的指标
    metrics_df = pd.DataFrame(metrics_list)
    
    # 保存整体评估指标
    metrics_df.to_csv(f"{output_dir}/all_window_metrics.csv", index=False)
    
    # 计算整体R2分数和其他指标
    all_predictions_np = np.array(all_predictions)
    all_actual_returns_np = np.array(all_actual_returns)
    all_valid_signals_np = np.array(all_valid_signals)
    
    overall_r2 = r2_score(all_actual_returns_np, all_predictions_np)
    
    # 只考虑有效信号
    if all_valid_signals_np.sum() > 0:
        valid_mse = mean_squared_error(
            all_actual_returns_np[all_valid_signals_np], 
            all_predictions_np[all_valid_signals_np]
        )
        valid_r2 = r2_score(
            all_actual_returns_np[all_valid_signals_np],
            all_predictions_np[all_valid_signals_np]
        )
        
        # 计算有效信号的方向准确率
        pred_direction = np.sign(all_predictions_np[all_valid_signals_np])
        true_direction = np.sign(all_actual_returns_np[all_valid_signals_np])
        valid_direction_accuracy = accuracy_score(true_direction, pred_direction)
        
        # 计算有效信号的IC
        valid_ic = np.corrcoef(all_predictions_np[all_valid_signals_np], all_actual_returns_np[all_valid_signals_np])[0, 1]
        
        # 真实收益率的平均值 - 信号效率衡量
        up_mask = all_predictions_np[all_valid_signals_np] > 0
        down_mask = all_predictions_np[all_valid_signals_np] < 0
        
        up_returns = all_actual_returns_np[all_valid_signals_np][up_mask].mean() if up_mask.sum() > 0 else 0
        down_returns = all_actual_returns_np[all_valid_signals_np][down_mask].mean() if down_mask.sum() > 0 else 0
        
        signal_efficiency = (up_returns - down_returns) if (up_mask.sum() > 0 and down_mask.sum() > 0) else np.nan
    else:
        valid_mse = np.nan
        valid_r2 = np.nan
        valid_direction_accuracy = np.nan
        valid_ic = np.nan
        signal_efficiency = np.nan
    
    # 计算所有信号的IC
    all_ic = np.corrcoef(all_predictions_np, all_actual_returns_np)[0, 1] if len(all_predictions_np) > 1 else np.nan
    
    overall_metrics = {
        'r2_score': overall_r2,
        'valid_signal_ratio': all_valid_signals_np.mean(),
        'valid_signal_mse': valid_mse,
        'valid_signal_r2': valid_r2,
        'valid_direction_accuracy': valid_direction_accuracy,
        'all_ic': all_ic,
        'valid_ic': valid_ic,
        'signal_efficiency': signal_efficiency
    }
    
    pd.DataFrame([overall_metrics]).to_csv(f"{output_dir}/overall_metrics_summary.csv", index=False)
    
    # 保存所有预测结果
    all_results_df = pd.DataFrame({
        'TradDay': all_dates,
        'InstruID': all_instruments,
        'actual_return': all_actual_returns,
        'original_prediction': all_predictions,  # 原始预测收益率
        'probability': all_probabilities,  # 映射后的概率值
        'filtered_prediction': np.where(all_valid_signals_np, all_predictions_np, 0),  # 过滤后的预测 (弱信号置为0)
        'signal_type': all_signal_types,  # 1=涨, -1=跌, 0=无信号
        'is_valid_signal': all_valid_signals  # 是否有效信号
    })
    
    all_results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    # 特征重要性
    if feature_importance_list:
        importance_df = pd.DataFrame(feature_importance_list)
        
        # 计算每个特征的平均重要性
        feature_stats = importance_df.groupby('feature').agg({
            'importance': 'mean',
            'used': 'sum'
        }).reset_index()
        
        # 计算使用率
        total_windows = len(metrics_list)
        feature_stats['usage_pct'] = feature_stats['used'] / total_windows * 100
        
        # 重命名列
        feature_stats.rename(columns={'importance': 'importance_mean'}, inplace=True)
        
        # 按重要性排序
        feature_stats = feature_stats.sort_values('importance_mean', ascending=False)
        
        # 保存特征重要性
        feature_stats.to_csv(f"{output_dir}/avg_feature_importance.csv", index=False)
    else:
        feature_stats = pd.DataFrame()
    
    # 创建最终结果字典
    results = {
        'metrics': metrics_df,
        'overall_metrics': overall_metrics,
        'feature_importance': feature_stats,
        'all_predictions': {
            'TradDay': all_dates,
            'InstruID': all_instruments,
            'actuals': all_actual_returns,
            'predictions': all_predictions,
            'valid_signals': all_valid_signals,
            'signal_type': all_signal_types
        }
    }
    
    return results

if __name__ == "__main__":
    main()
