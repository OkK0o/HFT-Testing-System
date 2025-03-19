import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def calculate_ic(predictions, returns):
    """计算IC值"""
    # 去除无效值
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    if not np.any(mask):
        print("警告: 计算IC时所有数据都是无效值")
        return np.nan
    
    predictions = predictions[mask]
    returns = returns[mask]
    
    if len(predictions) < 10:  # 最小样本量要求，调整为更小
        print(f"警告: 有效样本数量不足 ({len(predictions)} < 10)，无法计算可靠的IC")
        return np.nan
    
    # 检查预测值和收益率的方差
    if np.std(predictions) < 1e-10 or np.std(returns) < 1e-10:
        print(f"警告: 预测值或收益率几乎是常数，无法计算IC (std_pred={np.std(predictions):.6f}, std_returns={np.std(returns):.6f})")
        return np.nan
    
    # 计算Spearman相关系数
    try:
        ic = stats.spearmanr(predictions, returns)[0]
        if np.isnan(ic):
            print("警告: 计算的IC为NaN")
        return ic
    except Exception as e:
        print(f"计算IC时发生错误: {e}")
        return np.nan

def analyze_signal_distribution(predictions, returns):
    """分析信号分布和统计特征"""
    # 去除无效值
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    predictions = predictions[mask]
    returns = returns[mask]
    
    # 基本统计量
    stats_dict = {
        '信号均值': np.mean(predictions),
        '信号标准差': np.std(predictions),
        '信号偏度': stats.skew(predictions),
        '信号峰度': stats.kurtosis(predictions),
        '信号中位数': np.median(predictions),
        '信号最大值': np.max(predictions),
        '信号最小值': np.min(predictions),
        '信号样本量': len(predictions)
    }
    
    # 分位数分布
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_dict = {f'信号{q*100}分位数': np.percentile(predictions, q*100) for q in quantiles}
    stats_dict.update(quantile_dict)
    
    # 信号自相关性
    autocorr = pd.Series(predictions).autocorr()
    stats_dict['信号自相关性'] = autocorr
    
    # 信号与收益率的联合分布特征
    stats_dict['信号-收益率相关系数'] = stats.pearsonr(predictions, returns)[0]
    stats_dict['信号-收益率Spearman相关系数'] = stats.spearmanr(predictions, returns)[0]
    
    # 信号分布区间统计
    signal_ranges = np.percentile(predictions, [0, 25, 50, 75, 100])
    for i in range(len(signal_ranges)-1):
        range_mask = (predictions >= signal_ranges[i]) & (predictions < signal_ranges[i+1])
        range_returns = returns[range_mask]
        stats_dict[f'区间{i+1}收益率均值'] = np.mean(range_returns)
        stats_dict[f'区间{i+1}收益率标准差'] = np.std(range_returns)
        stats_dict[f'区间{i+1}样本量'] = np.sum(range_mask)
    
    return stats_dict

def train_xgboost_with_bayesian(df, 
                              target_col='10period_return', 
                              feature_cols=None, 
                              train_days=3,
                              val_days=1,
                              test_days=1,
                              min_train_samples=1000,
                              n_bayesian_iter=50,
                              output_dir="xgboost_bayesian_results",
                              standardize=True):
    """
    使用贝叶斯优化训练XGBoost模型，采用滚动窗口方式
    
    Args:
        df: 包含特征和标签的DataFrame
        target_col: 目标列名，默认为10period_return
        feature_cols: 特征列列表
        train_days: 训练窗口天数
        val_days: 验证窗口天数
        test_days: 测试窗口天数
        min_train_samples: 最小训练样本数
        n_bayesian_iter: 贝叶斯优化迭代次数
        output_dir: 输出目录
        standardize: 是否对特征进行标准化处理，默认为True
    
    Returns:
        包含模型、预测结果和评估指标的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
    
    # 目标变量放大以提高数值精度（例如放大10000倍）
    # 这样可以避免过小的MSE被舍入为0
    scale_factor = 10000
    print(f"将目标变量 {target_col} 放大 {scale_factor} 倍以提高数值精度")
    df[f'{target_col}_scaled'] = df[target_col] * scale_factor
    target_col_scaled = f'{target_col}_scaled'
    
    # 准备特征和标签
    y = df[target_col_scaled]
    X = df[feature_cols]
    date = pd.to_datetime(df['TradDay'])
    
    # 计算标签和特征的相关性
    print("计算特征与目标变量的相关性...")
    correlations = {}
    for feature in feature_cols:
        # 使用皮尔逊相关系数
        corr = pd.DataFrame({feature: X[feature], 'target': y}).corr().iloc[0, 1]
        correlations[feature] = corr
    
    # 特征预筛选 - 选择相关性最强的前200个特征
    max_features = min(200, len(feature_cols))
    top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:max_features]
    selected_features = [f[0] for f in top_features]
    print(f"预筛选 {len(selected_features)}/{len(feature_cols)} 个特征 (基于相关性)")
    
    # 更新特征集
    X = X[selected_features]
    feature_cols = selected_features
    
    # 确定唯一交易日
    unique_days = sorted(date.unique())
    print(f"数据集包含 {len(unique_days)} 个交易日")
    
    # 初始化结果存储
    all_predictions = pd.Series(index=df.index, dtype=float)
    all_metrics = []
    window_results = []
    
    # 滚动窗口训练
    window_size = train_days + val_days + test_days
    
    for start_idx in range(0, len(unique_days) - window_size + 1):
        window_days = unique_days[start_idx:start_idx + window_size]
        train_window = window_days[:train_days]
        val_window = window_days[train_days:train_days + val_days]
        test_window = window_days[train_days + val_days:]
        
        print(f"\n=== 窗口 {start_idx+1}/{len(unique_days) - window_size + 1} ===")
        print(f"训练日期: {min(train_window)} 至 {max(train_window)}")
        print(f"验证日期: {min(val_window)} 至 {max(val_window)}")
        print(f"测试日期: {min(test_window)} 至 {max(test_window)}")
        
        # 创建训练、验证和测试掩码
        train_mask = date.isin(train_window)
        val_mask = date.isin(val_window)
        test_mask = date.isin(test_window)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 保存未缩放的原始测试标签（用于最终评估）
        y_test_original = df.loc[test_mask, target_col]
        
        print(f"训练样本: {X_train.shape[0]}")
        print(f"验证样本: {X_val.shape[0]}")
        print(f"测试样本: {X_test.shape[0]}")
        
        # 如果训练样本不足，跳过此窗口
        if len(X_train) < min_train_samples:
            print(f"警告: 训练样本不足 ({len(X_train)} < {min_train_samples})，跳过此窗口")
            continue
            
        # 特征标准化 - 使用训练集的均值和标准差对所有数据进行标准化
        if standardize:
            print("对特征进行标准化处理...")
            scaler = StandardScaler()
            
            # 注意：我们只使用训练集来拟合scaler，然后应用到所有数据集
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            
            # 使用训练集拟合的scaler来变换验证集和测试集
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
            
            # 输出标准化前后的统计信息用于验证
            print("标准化前的特征统计 (训练集):")
            print(f"  均值范围: [{X_train.mean().min():.4f}, {X_train.mean().max():.4f}]")
            print(f"  标准差范围: [{X_train.std().min():.4f}, {X_train.std().max():.4f}]")
            
            print("标准化后的特征统计 (训练集):")  
            print(f"  均值范围: [{X_train_scaled.mean().min():.4f}, {X_train_scaled.mean().max():.4f}]")
            print(f"  标准差范围: [{X_train_scaled.std().min():.4f}, {X_train_scaled.std().max():.4f}]")
            
            print("标准化后的特征统计 (测试集):")
            print(f"  均值范围: [{X_test_scaled.mean().min():.4f}, {X_test_scaled.mean().max():.4f}]")
            print(f"  标准差范围: [{X_test_scaled.std().min():.4f}, {X_test_scaled.std().max():.4f}]")
            
            # 将标准化后的数据作为训练数据
            X_train = X_train_scaled
            X_val = X_val_scaled
            X_test = X_test_scaled
            
            # 保存特征均值和标准差，用于后续分析和预测
            scaler_output = pd.DataFrame({
                'feature': X_train.columns,
                'mean': scaler.mean_,
                'scale': scaler.scale_
            })
            
            # 创建scaler输出目录
            window_model_dir = f"{output_dir}/window_{start_idx+1}"
            os.makedirs(window_model_dir, exist_ok=True)
            scaler_output.to_csv(f"{window_model_dir}/feature_scaler.csv", index=False)
        
        # 特征选择 - 在每个窗口上动态选择特征
        from sklearn.feature_selection import SelectFromModel
        
        # 使用简单的XGBoost模型进行特征重要性筛选
        feature_selector = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        feature_selector.fit(X_train, y_train)
        
        # 选择重要特征
        selection = SelectFromModel(feature_selector, prefit=True, threshold='median')
        feature_mask = selection.get_support()
        window_features = [feature_cols[i] for i in range(len(feature_cols)) if feature_mask[i]]
        n_selected = len(window_features)
        print(f"为此窗口选择了 {n_selected}/{len(feature_cols)} 个特征")
        
        # 如果筛选后特征太少，使用全部特征
        if n_selected < 10:
            print("选择的特征太少，使用所有特征")
            window_features = feature_cols
        
        # 更新训练数据
        X_train = X_train[window_features]
        X_val = X_val[window_features]
        X_test = X_test[window_features]
        
        # 创建DMatrix数据
        dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        dval = xgb.DMatrix(X_val.values, label=y_val.values)
        dtest = xgb.DMatrix(X_test.values, label=y_test.values)
        
        # 定义参数空间 - 扩展范围和种类
        param_space = {
            'max_depth': (3, 10),                      # 树的最大深度
            'learning_rate': (0.005, 0.3),             # 学习率 - 降低下限
            'subsample': (0.5, 1.0),                   # 样本采样比例
            'colsample_bytree': (0.5, 1.0),            # 特征采样比例
            'min_child_weight': (1, 20),               # 最小叶子节点样本权重和 - 增大上限
            'gamma': (0, 10),                          # 节点分裂所需的最小损失函数减少值 - 增大上限
            'alpha': (0, 10),                          # L1正则化项 (新增)
            'lambda': (1, 10)                          # L2正则化项 (新增)
        }
        
        # 设置基础参数
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],  # 添加多个评估指标
            'seed': 42
        }
            
        # 定义评估函数
        def objective(params):
            # 合并基础参数和贝叶斯优化参数
            current_params = base_params.copy()
            current_params['max_depth'] = int(params[0])
            current_params['learning_rate'] = params[1]
            current_params['subsample'] = params[2]
            current_params['colsample_bytree'] = params[3]
            current_params['min_child_weight'] = int(params[4])
            current_params['gamma'] = params[5]
            current_params['alpha'] = params[6]
            current_params['lambda'] = params[7]
            
            # 评估模型
            evaluation_results = {}
            try:
                model = xgb.train(
                    current_params,
                    dtrain,
                    num_boost_round=300,  # 增加迭代次数
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=30,  # 增加早停轮数
                    evals_result=evaluation_results,
                    verbose_eval=False
                )
                # 返回最佳迭代的验证集RMSE
                val_rmse = evaluation_results['val']['rmse'][model.best_iteration]
                return val_rmse
            except Exception as e:
                print(f"参数评估失败: {e}")
                return float('inf')  # 返回一个很大的错误值
        
        # 使用skopt的贝叶斯优化
        from skopt import gp_minimize
        from skopt.utils import use_named_args
        from skopt.space import Real, Integer
        
        # 创建正确的搜索空间
        dimensions = [
            Integer(3, 10, name='max_depth'),
            Real(0.005, 0.3, prior='log-uniform', name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Integer(1, 20, name='min_child_weight'),
            Real(0, 10, name='gamma'),
            Real(0, 10, name='alpha'),
            Real(1, 10, name='lambda')
        ]
        
        # 设置初始点
        init_points = [
            [6, 0.1, 0.8, 0.8, 1, 0, 0, 1],     # 默认参数
            [8, 0.05, 0.9, 0.7, 3, 1, 1, 1],    # 另一组参数
            [5, 0.02, 0.7, 0.9, 5, 2, 2, 2],    # 第三组参数 (新增)
            [4, 0.01, 0.6, 0.6, 10, 5, 5, 5]    # 第四组参数 (新增)
        ]
        
        print("开始贝叶斯参数调优...")
        
        try:
            # 运行贝叶斯优化
            result = gp_minimize(
                objective,
                dimensions=dimensions,
                n_calls=n_bayesian_iter,
                n_initial_points=min(4, n_bayesian_iter),
                x0=init_points,
                random_state=42,
                verbose=False
            )
            
            # 获取最佳参数
            best_params = {
                'max_depth': int(result.x[0]),
                'learning_rate': result.x[1],
                'subsample': result.x[2],
                'colsample_bytree': result.x[3],
                'min_child_weight': int(result.x[4]),
                'gamma': result.x[5],
                'alpha': result.x[6],
                'lambda': result.x[7],
                'n_estimators': 2000  # 增加估计器数量
            }
            print(f"找到最佳参数: {best_params}")
            best_rmse = result.fun
            print(f"最佳验证RMSE: {best_rmse:.6f}")
            
        except Exception as e:
            print(f"贝叶斯优化失败，使用默认参数: {e}")
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,  # 降低默认学习率
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,  # 增加默认值
                'gamma': 1,  # 增加默认值
                'alpha': 0,
                'lambda': 1,
                'n_estimators': 200
            }
        
        # 使用最佳参数训练最终模型
        print("使用最佳参数训练最终模型...")
        
        # 设置参数
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'seed': 42,
        }
        
        # 从best_params中复制参数
        for k, v in best_params.items():
            if k != 'n_estimators':
                params[k] = v
        
        # 训练模型
        eval_list = [(dtrain, 'train'), (dval, 'validation')]
        
        # 使用原生XGBoost接口训练模型
        best_model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=2000,  # 增加迭代次数，由early_stopping控制实际迭代次数
            evals=eval_list, 
            early_stopping_rounds=50,
            verbose_eval=100  # 每100轮打印一次
        )
        
        print(f"最佳迭代次数: {best_model.best_iteration}")
        
        # 保存每个窗口的模型
        window_model_dir = f"{output_dir}/window_{start_idx+1}"
        os.makedirs(window_model_dir, exist_ok=True)
        best_model.save_model(f"{window_model_dir}/xgb_model.json")
        joblib.dump(best_params, f"{window_model_dir}/best_params.pkl")
        
        # 在测试集上进行预测 (使用DMatrix)
        test_pred_scaled = best_model.predict(dtest, iteration_range=(0, best_model.best_iteration + 1))
        
        # 将预测值缩放回原始尺度
        test_pred = test_pred_scaled / scale_factor
        
        # 打印预测值统计信息，用于调试
        print(f"预测值统计 (原始尺度): min={np.min(test_pred):.6f}, max={np.max(test_pred):.6f}, mean={np.mean(test_pred):.6f}, std={np.std(test_pred):.6f}")
        print(f"真实值统计: min={np.min(y_test_original):.6f}, max={np.max(y_test_original):.6f}, mean={np.mean(y_test_original):.6f}, std={np.std(y_test_original):.6f}")
        
        # 如果预测值或真实值都是常数，会导致指标为NaN
        if np.std(test_pred) < 1e-10:
            print("警告: 预测值几乎是常数，增加特征多样性或调整模型参数")
            # 添加少量噪声以避免常数预测
            test_pred = test_pred + np.random.normal(0, 1e-6, len(test_pred))
        
        # 存储测试集预测
        test_indices = df.index[test_mask]
        all_predictions.loc[test_indices] = test_pred
        
        # 计算评估指标
        # 检查是否有足够的数据计算指标
        if len(y_test_original) > 0 and not np.all(np.isnan(y_test_original)) and not np.all(np.isnan(test_pred)):
            # 使用原始尺度计算指标
            test_mse = mean_squared_error(y_test_original, test_pred)
            test_mae = mean_absolute_error(y_test_original, test_pred)
            test_r2 = r2_score(y_test_original, test_pred)
            test_ic = calculate_ic(test_pred, y_test_original)
            
            # 添加自定义评估指标：预测收益率符号正确的平均收益
            valid_mask = ~(np.isnan(y_test_original) | np.isnan(test_pred))
            y_valid = y_test_original[valid_mask]
            pred_valid = test_pred[valid_mask]
            
            # 计算符号预测正确的样本
            correct_sign = np.sign(y_valid) == np.sign(pred_valid)
            if np.sum(correct_sign) > 0:
                avg_return_correct = np.mean(np.abs(y_valid[correct_sign]))
            else:
                avg_return_correct = 0
                
            # 添加更详细的MSE分析
            print(f"MSE详细分析:")
            print(f"  - 原始MSE值 (高精度): {test_mse:.15f}")
            print(f"  - 预测值的均值: {np.mean(test_pred):.10f}, 方差: {np.var(test_pred):.10f}")
            print(f"  - 实际值的均值: {np.mean(y_test_original):.10f}, 方差: {np.var(y_test_original):.10f}")
            
            # 计算预测误差统计
            errors = y_test_original - test_pred
            print(f"  - 预测误差统计: 均值={np.mean(errors):.10f}, 方差={np.var(errors):.10f}, 最大={np.max(np.abs(errors)):.10f}")
            
            # 检查是否有异常预测
            extreme_errors = np.abs(errors) > 0.001  # 定义为超过0.1%的预测误差
            if np.any(extreme_errors):
                print(f"  - 发现 {np.sum(extreme_errors)} 个异常预测 (超过0.1%误差)")
            
            # 计算方向准确率前检查符号
            valid_mask = ~(np.isnan(y_test_original) | np.isnan(test_pred))
            if np.sum(valid_mask) > 0:
                test_direction_accuracy = np.mean(np.sign(y_test_original[valid_mask]) == np.sign(test_pred[valid_mask]))
            else:
                test_direction_accuracy = np.nan
            
            # 计算盈亏比
            y_valid = y_test_original[valid_mask]
            pred_valid = test_pred[valid_mask]
            if np.sum(valid_mask) > 0:
                correct_pred = (y_valid * pred_valid > 0)
                if np.sum(correct_pred) > 0 and np.sum(~correct_pred) > 0:
                    test_profits = np.sum(np.abs(y_valid[correct_pred]))
                    test_losses = np.sum(np.abs(y_valid[~correct_pred]))
                    test_profit_loss_ratio = test_profits / test_losses if test_losses > 0 else np.nan
                else:
                    test_profit_loss_ratio = np.nan
            else:
                test_profit_loss_ratio = np.nan
        else:
            print("警告: 测试集为空或者全是NaN，无法计算评估指标")
            test_mse = test_mae = test_r2 = test_ic = test_direction_accuracy = test_profit_loss_ratio = avg_return_correct = np.nan
        
        window_metric = {
            'window': start_idx + 1,
            'train_start': min(train_window),
            'train_end': max(train_window),
            'test_start': min(test_window),
            'test_end': max(test_window),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2,
            'ic': test_ic,
            'direction_accuracy': test_direction_accuracy,
            'profit_loss_ratio': test_profit_loss_ratio,
            'avg_return_correct_sign': avg_return_correct
        }
        
        all_metrics.append(window_metric)
        
        # 保存窗口结果
        window_results.append({
            'window': start_idx + 1,
            'model': best_model,
            'best_params': best_params,
            'features_used': window_features,
            'metrics': window_metric,
            'feature_importance': best_model.get_score(importance_type='gain')
        })
        
        # 打印当前窗口评估结果
        print(f"测试集评估指标:")
        print(f"MSE: {test_mse:.10f}, MAE: {test_mae:.10f}")
        print(f"R2: {test_r2:.6f}, IC: {test_ic:.6f}")
        print(f"方向准确率: {test_direction_accuracy:.6f}")
        print(f"盈亏比: {test_profit_loss_ratio:.6f}")
        print(f"符号正确时的平均收益: {avg_return_correct:.10f}")
    
    # 没有成功训练任何窗口
    if not all_metrics:
        print("错误: 没有足够数据进行训练")
        return None
    
    # 保存所有窗口评估指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{output_dir}/all_window_metrics.csv", index=False)
    
    # 计算整体评估指标
    valid_mask = ~all_predictions.isna()
    valid_returns = df.loc[valid_mask, target_col]
    
    # 使用原始尺度计算整体指标
    overall_mse = mean_squared_error(valid_returns, all_predictions[valid_mask])
    overall_r2 = r2_score(valid_returns, all_predictions[valid_mask])
    overall_ic = calculate_ic(all_predictions[valid_mask], valid_returns)
    overall_direction_accuracy = np.mean(np.sign(valid_returns) == np.sign(all_predictions[valid_mask]))
    
    # 计算夏普比率
    pred_sign = np.sign(all_predictions[valid_mask])
    strategy_returns = pred_sign * valid_returns
    daily_returns = strategy_returns.groupby(df.loc[valid_mask, 'TradDay']).mean()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else np.nan
    
    # 计算信息比率
    benchmark_returns = valid_returns  # 假设基准是买入持有
    daily_excess_returns = (strategy_returns - benchmark_returns).groupby(df.loc[valid_mask, 'TradDay']).mean()
    information_ratio = np.sqrt(252) * daily_excess_returns.mean() / daily_excess_returns.std() if daily_excess_returns.std() > 0 else np.nan
    
    # 计算最大回撤
    cum_returns = daily_returns.cumsum()
    max_drawdown = 0
    if len(cum_returns) > 0:
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    
    # 创建汇总统计表格
    print("创建窗口指标汇总统计表...")
    # 从metrics_df中获取窗口指标
    summary_stats = {
        'MSE': metrics_df['mse'].describe(),
        'MAE': metrics_df['mae'].describe(),
        'R2': metrics_df['r2'].describe(),
        'IC': metrics_df['ic'].describe(),
        'Direction_Accuracy': metrics_df['direction_accuracy'].describe(),
        'Profit_Loss_Ratio': metrics_df['profit_loss_ratio'].describe(),
    }
    
    # 创建窗口汇总表格
    stats_df = pd.DataFrame()
    for metric, stats in summary_stats.items():
        stats_df[metric] = stats
    
    # 添加整体指标
    overall_metrics = {
        'overall_mse': overall_mse,
        'overall_r2': overall_r2,
        'overall_ic': overall_ic,
        'overall_direction_accuracy': overall_direction_accuracy,
        'overall_sharpe_ratio': sharpe_ratio,
        'overall_information_ratio': information_ratio,
        'overall_max_drawdown': max_drawdown,
        'total_windows': len(all_metrics),
        'windows_with_positive_ic': (metrics_df['ic'] > 0).sum(),
        'windows_with_accuracy_above_50': (metrics_df['direction_accuracy'] > 0.5).sum(),
        'percent_windows_with_positive_ic': (metrics_df['ic'] > 0).mean() * 100,
        'percent_windows_with_accuracy_above_50': (metrics_df['direction_accuracy'] > 0.5).mean() * 100,
    }
    
    # 保存汇总统计
    stats_df.to_csv(f"{output_dir}/window_metrics_summary.csv")
    pd.Series(overall_metrics).to_csv(f"{output_dir}/overall_metrics_summary.csv")
    
    # 创建详细的窗口指标表格
    detailed_window_metrics = metrics_df.copy()
    
    # 添加窗口特定信息
    for i, result in enumerate(window_results):
        if i < len(detailed_window_metrics):
            # 添加使用的特征数量
            detailed_window_metrics.loc[i, 'num_features_used'] = len(result['features_used'])
            
            # 添加模型参数
            for param, value in result['best_params'].items():
                detailed_window_metrics.loc[i, f'param_{param}'] = value
            
            # 添加迭代次数
            if hasattr(result['model'], 'best_iteration'):
                detailed_window_metrics.loc[i, 'best_iteration'] = result['model'].best_iteration
    
    # 保存详细窗口指标
    detailed_window_metrics.to_csv(f"{output_dir}/detailed_window_metrics.csv", index=False)
    
    # 添加参数与性能相关性分析
    print("分析模型参数与性能指标的相关性...")
    
    # 提取参数列
    param_cols = [col for col in detailed_window_metrics.columns if col.startswith('param_')]
    
    if len(param_cols) > 0:
        # 主要性能指标
        perf_cols = ['mse', 'r2', 'ic', 'direction_accuracy', 'profit_loss_ratio']
        
        # 计算每个参数与性能指标的相关性
        corr_data = []
        for param in param_cols:
            param_name = param.replace('param_', '')
            for perf in perf_cols:
                # 计算Spearman相关系数
                mask = ~detailed_window_metrics[param].isna() & ~detailed_window_metrics[perf].isna()
                if mask.sum() > 5:  # 至少需要5个非NA值
                    corr, p_value = stats.spearmanr(
                        detailed_window_metrics.loc[mask, param], 
                        detailed_window_metrics.loc[mask, perf]
                    )
                    corr_data.append({
                        'Parameter': param_name,
                        'Metric': perf,
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05
                    })
        
        # 创建相关性表格
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            corr_df.to_csv(f"{output_dir}/param_performance_correlation.csv", index=False)
            
            # 创建重要参数可视化
            significant_corrs = corr_df[corr_df['Significant']]
            if len(significant_corrs) > 0:
                plt.figure(figsize=(12, 8))
                
                # 根据相关性绝对值排序
                significant_corrs['AbsCorr'] = significant_corrs['Correlation'].abs()
                significant_corrs = significant_corrs.sort_values('AbsCorr', ascending=False).head(20)
                
                # 创建热力图数据
                pivot_data = significant_corrs.pivot_table(
                    index='Parameter', 
                    columns='Metric', 
                    values='Correlation',
                    aggfunc='first'
                ).fillna(0)
                
                # 绘制热力图
                sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Significant Parameter-Performance Correlations')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/param_performance_heatmap.png", dpi=300)
    
    # 创建分类表格 - 按性能分组分析窗口
    print("按性能分组分析窗口...")
    
    # 根据IC值将窗口分为高、中、低三组
    metrics_df['ic_group'] = pd.qcut(metrics_df['ic'], 3, labels=['Low', 'Medium', 'High'])
    
    # 按组计算特征数量和参数平均值
    group_stats = []
    for name, group in metrics_df.groupby('ic_group'):
        if len(group) > 0:
            # 窗口数量和性能指标
            stat = {
                'Group': name,
                'Count': len(group),
                'Avg_IC': group['ic'].mean(),
                'Avg_Direction_Accuracy': group['direction_accuracy'].mean(),
                'Avg_R2': group['r2'].mean()
            }
            
            # 添加窗口索引
            stat['Windows'] = ', '.join(map(str, group['window'].tolist()))
            
            group_stats.append(stat)
    
    # 创建分组统计表格
    if group_stats:
        group_df = pd.DataFrame(group_stats)
        group_df.to_csv(f"{output_dir}/performance_group_analysis.csv", index=False)
    
    print(f"\n=== 整体评估指标 ===")
    print(f"窗口数量: {len(all_metrics)}")
    print(f"覆盖样本数: {np.sum(valid_mask)}")
    print(f"整体 MSE (高精度): {overall_mse:.15f}")
    print(f"整体 R2: {overall_r2:.6f}")
    print(f"整体 IC: {overall_ic:.6f}")
    print(f"整体方向准确率: {overall_direction_accuracy:.6f}")
    print(f"策略夏普比率: {sharpe_ratio:.6f}")
    print(f"策略信息比率: {information_ratio:.6f}")
    
    # 添加整体误差分析
    all_errors = valid_returns - all_predictions[valid_mask]
    print(f"整体误差统计:")
    print(f"  - 平均误差: {np.mean(all_errors):.10f}")
    print(f"  - 误差方差: {np.var(all_errors):.10f}")
    print(f"  - 最大绝对误差: {np.max(np.abs(all_errors)):.10f}")
    print(f"  - 误差分布中位数: {np.median(all_errors):.10f}")
    
    # 添加预测与实际值相关性分析
    print(f"预测与实际值相关性:")
    print(f"  - 皮尔逊相关系数: {stats.pearsonr(all_predictions[valid_mask], valid_returns)[0]:.6f}")
    print(f"  - 斯皮尔曼相关系数: {stats.spearmanr(all_predictions[valid_mask], valid_returns)[0]:.6f}")
    
    # 平均特征重要性
    # 因为使用了原生XGBoost接口，特征重要性的获取方式需要调整
    all_feature_importance = {}
    all_feature_usage = {}  # 记录每个特征被使用的次数
    
    # 收集所有窗口的特征重要性
    for result in window_results:
        importance_dict = result['feature_importance']
        used_features = result['features_used']
        
        # 更新特征使用次数
        for feature in used_features:
            if feature not in all_feature_usage:
                all_feature_usage[feature] = 0
            all_feature_usage[feature] += 1
        
        # 更新特征重要性
        for feature, importance in importance_dict.items():
            if feature not in all_feature_importance:
                all_feature_importance[feature] = []
            all_feature_importance[feature].append(importance)
    
    # 计算平均特征重要性
    avg_feature_importance = {}
    for feature, values in all_feature_importance.items():
        avg_feature_importance[feature] = np.mean(values)
    
    # 创建包含使用次数的特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': list(avg_feature_importance.keys()),
        'Importance': list(avg_feature_importance.values()),
        'Usage_Count': [all_feature_usage.get(f, 0) for f in avg_feature_importance.keys()],
        'Usage_Percentage': [all_feature_usage.get(f, 0) / len(window_results) * 100 for f in avg_feature_importance.keys()]
    }).sort_values('Importance', ascending=False)
    
    feature_importance_df.to_csv(f"{output_dir}/avg_feature_importance.csv", index=False)
    
    # 可视化平均特征重要性（前20个）和使用频率
    plt.figure(figsize=(14, 12))
    top_n = min(20, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    
    # 创建颜色映射，基于使用频率
    norm = plt.Normalize(top_features['Usage_Percentage'].min(), top_features['Usage_Percentage'].max())
    colors = plt.cm.viridis(norm(top_features['Usage_Percentage']))
    
    # 绘制特征重要性条形图
    bars = plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
    
    # 添加使用频率标签
    for i, (_, row) in enumerate(top_features.iterrows()):
        plt.text(row['Importance'] + row['Importance']*0.01, i, 
                 f"{row['Usage_Percentage']:.1f}%", 
                 va='center', fontsize=9)
    
    plt.title('Top 20 Features by Importance (color = usage frequency)')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), 
                 label='Usage Percentage (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_feature_importance.png", dpi=300)
    
    # 可视化每个窗口的IC值和方向准确率
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    ic_values = metrics_df['ic']
    plt.bar(range(len(metrics_df)), ic_values, alpha=0.7, color='royalblue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=ic_values.mean(), color='g', linestyle='-', 
                label=f'平均 IC: {ic_values.mean():.4f}')
    plt.title('各窗口IC值')
    plt.ylabel('Information Coefficient')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    dir_acc = metrics_df['direction_accuracy']
    plt.bar(range(len(metrics_df)), dir_acc, alpha=0.7, color='darkorange')
    plt.axhline(y=0.5, color='r', linestyle='--', label='随机水平 (50%)')
    plt.axhline(y=dir_acc.mean(), color='g', linestyle='-', 
                label=f'平均方向准确率: {dir_acc.mean():.4f}')
    plt.title('各窗口方向准确率')
    plt.xlabel('窗口索引')
    plt.ylabel('方向准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/window_ic_and_direction.png", dpi=300)
    
    # 绘制R2和MSE趋势图
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    r2_values = metrics_df['r2']
    plt.plot(range(len(metrics_df)), r2_values, 'o-', alpha=0.7, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=r2_values.mean(), color='g', linestyle='-', 
                label=f'平均 R2: {r2_values.mean():.4f}')
    plt.title('各窗口R²值')
    plt.ylabel('R²')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    mse_values = metrics_df['mse']
    plt.semilogy(range(len(metrics_df)), mse_values, 'o-', alpha=0.7, color='teal')
    plt.axhline(y=mse_values.mean(), color='g', linestyle='-', 
                label=f'平均 MSE: {mse_values.mean():.10f}')
    plt.title('各窗口MSE值 (对数尺度)')
    plt.xlabel('窗口索引')
    plt.ylabel('MSE (log scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/window_r2_and_mse.png", dpi=300)
    
    # 绘制预测值与实际值的散点图 (添加密度等高线)
    plt.figure(figsize=(10, 8))
    
    # 创建散点图
    plt.scatter(all_predictions[valid_mask], valid_returns, alpha=0.3, s=10, color='blue')
    
    # 添加密度等高线
    try:
        from scipy.stats import gaussian_kde
        
        # 计算点密度
        xy = np.vstack([all_predictions[valid_mask], valid_returns])
        density = gaussian_kde(xy)(xy)
        
        # 根据密度排序点
        idx = density.argsort()
        x, y, z = np.array(all_predictions[valid_mask])[idx], np.array(valid_returns)[idx], density[idx]
        
        # 绘制密度散点图
        plt.scatter(x, y, c=z, s=15, alpha=0.5, cmap='viridis')
        plt.colorbar(label='点密度')
    except Exception as e:
        print(f"绘制密度图失败: {e}")
    
    # 添加拟合线
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(all_predictions[valid_mask], valid_returns)
        plt.plot(np.sort(all_predictions[valid_mask]), 
                intercept + slope * np.sort(all_predictions[valid_mask]), 
                'r-', linewidth=2, 
                label=f'拟合线 (r={r_value:.4f})')
    except Exception as e:
        print(f"绘制拟合线失败: {e}")
    
    plt.xlabel('预测值')
    plt.ylabel('实际收益率')
    plt.title('预测值 vs 实际收益率')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_vs_actual.png", dpi=300)
    
    # 绘制分位数分析图
    plt.figure(figsize=(12, 8))
    
    # 将预测分成10个分位数
    pred_df = pd.DataFrame({'pred': all_predictions[valid_mask], 'actual': valid_returns})
    pred_df['quantile'] = pd.qcut(pred_df['pred'], 10, labels=False)
    
    # 计算每个分位数的平均实际收益率
    quantile_returns = pred_df.groupby('quantile')['actual'].mean()
    
    # 绘制每个分位数的平均收益率
    plt.bar(range(10), quantile_returns, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(range(10), [f'Q{i+1}' for i in range(10)])
    plt.xlabel('预测值分位数（从低到高）')
    plt.ylabel('平均实际收益率')
    plt.title('预测分位数收益分析')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quantile_returns.png", dpi=300)
    
    # 保存完整预测结果
    prediction_df = pd.DataFrame({
        'datetime': df['DateTime'],
        'tradday': df['TradDay'],
        'instruID': df['InstruID'],
        'actual_return': df[target_col],
        'prediction': all_predictions
    })
    prediction_df.to_csv(f"{output_dir}/all_predictions.csv", index=False)
    
    # 分析信号分布
    signal_stats = analyze_signal_distribution(all_predictions[valid_mask], valid_returns)
    pd.Series(signal_stats).to_csv(f"{output_dir}/signal_stats.csv")
    
    # 汇总结果
    results = {
        'window_results': window_results,
        'predictions': all_predictions,
        'metrics': metrics_df,
        'overall_metrics': {
            'overall_mse': overall_mse,
            'overall_r2': overall_r2,
            'overall_ic': overall_ic,
            'direction_accuracy': overall_direction_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio
        },
        'signal_stats': signal_stats,
        'feature_importance': feature_importance_df,
        'predictions_df': prediction_df
    }
    
    return results

if __name__ == "__main__":
    # 加载因子调整后的数据
    print("加载数据...")
    df = pd.read_feather("data_with_factors_adjusted.feather")
    print(f"数据形状: {df.shape}")
    
    # 选择特征
    basic_columns = ['index', 'TradDay', 'UpdateTime', 'InstruID', 'LastPrice', 'HighPrice', 
                    'LowPrice', 'OpenPrice', 'Volume', 'LastVolume', 'Turnover', 'OpenInt', 
                    'PreOpenInt', 'OpenIntChg', 'ClosePrice', 'SetPrice', 'PreSetPrice', 
                    'PreCloPrice', 'BuyVolume', 'SellVolume', 'AvgBuyPrice', 'AvgSellPrice', 
                    'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 
                    'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5',
                    'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
                    'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
                    'DerBidVolume1', 'DerBidVolume2', 'DerBidVolume3', 'DerBidVolume4', 'DerBidVolume5',
                    'DerAskVolume1', 'DerAskVolume2', 'DerAskVolume3', 'DerAskVolume4', 'DerAskVolume5',
                    'ULimitPrice', 'LLimitPrice', 'InstruCode', 'DateTime', 'hour', 'minute', 'second',
                    'day_of_week', 'session', 'is_holiday', 'is_overnight', 'expiry_date', 'days_to_expiry',
                    'mid_price', 'returns', 'vol', 'turnover', 'vwap', 'spread', 'depth_imbalance',
                    '10period_return']  # 排除收益率列
    
    # 筛选因子列
    factor_columns = [col for col in df.columns if col not in basic_columns]
    print(f"发现 {len(factor_columns)} 个可用因子")
    
    # 排除vpin因子
    selected_factors = [factor for factor in factor_columns if 'vpin' not in factor]
    print(f"选择 {len(selected_factors)} 个因子进行训练 (排除vpin)")
    
    # 运行训练
    results = train_xgboost_with_bayesian(
        df=df,
        target_col='10period_return',
        feature_cols=selected_factors,
        min_train_samples=1000,
        n_bayesian_iter=50,
        output_dir="xgboost_bayesian_results"
    )
    
    print("\n训练完成! 所有结果已保存至 xgboost_bayesian_results 目录")