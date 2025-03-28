import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import spearmanr, pearsonr  # 明确导入pearsonr函数
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
import traceback

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
    stats_dict['信号-收益率Spearman相关系数'] = spearmanr(predictions, returns)[0]
    
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
    
    # 初始化特征重要性收集列表
    feature_importances = []
    
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
            
            # 金融时序数据的稳健标准化策略
            # 1. 进行特征稳定性分析
            print("进行特征稳定性分析...")
            feature_stability = {}
            
            # 如果训练窗口超过1天，分析每个特征在不同日期的变化
            if len(train_window) > 1:
                try:
                    train_daily_stats = df[df['TradDay'].isin(train_window)].groupby('TradDay')[feature_cols].mean()
                    # 计算变异系数(CV = std/mean) - 较低的CV表示更稳定的特征
                    feature_cv = {}
                    for feature in feature_cols:
                        mean_vals = train_daily_stats[feature]
                        cv = mean_vals.std() / mean_vals.mean() if mean_vals.mean() != 0 else float('inf')
                        feature_cv[feature] = abs(cv)  # 使用绝对值
                    
                    # 选择最稳定的特征
                    stable_threshold = np.percentile(list(feature_cv.values()), 75)  # 75%分位数
                    stable_features = [f for f, cv in feature_cv.items() if cv <= stable_threshold]
                    print(f"基于日间稳定性分析，选择了{len(stable_features)}/{len(feature_cols)}个相对稳定的特征")
                    
                    # 如果稳定特征太少，则使用所有特征
                    if len(stable_features) < 10:
                        print("稳定特征太少，使用所有特征")
                        stable_features = feature_cols
                except Exception as e:
                    print(f"特征稳定性分析失败: {e}，使用所有特征")
                    stable_features = feature_cols
            else:
                # 只有一天数据时，无法分析稳定性，使用所有特征
                stable_features = feature_cols
                
            # 2. 应用更稳健的标准化方法
            from sklearn.preprocessing import RobustScaler  # 稳健标准化，对异常值不敏感
            
            # 针对金融数据特点使用RobustScaler而非StandardScaler
            scaler = RobustScaler(quantile_range=(5, 95))  # 使用5%-95%分位数而非默认的25%-75%
            
            # 注意：我们只使用训练集来拟合scaler
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
            
            # 3. 处理标准化后的极端值（对测试集尤为重要）
            clip_threshold = 10.0  # 可调整的阈值
            for col in X_test_scaled.columns:
                # 只处理明显异常的值
                X_test_scaled[col] = X_test_scaled[col].clip(-clip_threshold, clip_threshold)
            
            # 根据需要处理验证集
            for col in X_val_scaled.columns:
                X_val_scaled[col] = X_val_scaled[col].clip(-clip_threshold, clip_threshold)
            
            # 输出标准化前后的统计信息用于验证
            print("标准化前后特征分布对比(简化版):")
            print(f"  训练集 - 标准化前均值范围: [{X_train.mean().min():.4f}, {X_train.mean().max():.4f}]")
            print(f"  训练集 - 标准化后均值范围: [{X_train_scaled.mean().min():.4f}, {X_train_scaled.mean().max():.4f}]")
            print(f"  测试集 - 标准化后均值范围: [{X_test_scaled.mean().min():.4f}, {X_test_scaled.mean().max():.4f}]")
            print(f"  测试集 - 标准化后标准差范围: [{X_test_scaled.std().min():.4f}, {X_test_scaled.std().max():.4f}]")
            
            # 保存标准化统计信息（简化版）
            window_model_dir = f"{output_dir}/window_{start_idx+1}"
            os.makedirs(window_model_dir, exist_ok=True)
            
            # 4. 减少图形生成，只在需要时生成
            # 只在第一个窗口和每10个窗口保存一次分布图
            if start_idx == 0 or (start_idx + 1) % 10 == 0:
                try:
                    # 选择最多3个有代表性的特征
                    n_features_to_plot = min(3, len(X_train.columns))
                    # 按方差排序选择最具变化性的特征
                    features_to_plot = X_train.var().sort_values(ascending=False).index[:n_features_to_plot]
                    
                    # 创建标准化前后分布对比图（简化版）
                    plt.figure(figsize=(12, 4*n_features_to_plot))
                    for i, feature in enumerate(features_to_plot):
                        plt.subplot(n_features_to_plot, 2, i*2+1)
                        sns.histplot(X_train[feature], kde=True, bins=30)
                        plt.title(f"{feature} - Before Standardization")
                        plt.xlabel("Value")
                        
                        plt.subplot(n_features_to_plot, 2, i*2+2)
                        sns.histplot(X_train_scaled[feature], kde=True, bins=30)
                        plt.title(f"{feature} - After Standardization")
                        plt.xlabel("Value")
                    
                    plt.tight_layout()
                    plt.savefig(f"{window_model_dir}/feature_distribution_comparison.png", dpi=100)
                    plt.close()  # 关闭图表释放内存
                except Exception as e:
                    print(f"绘制特征分布图失败: {e}")
            
            # 将标准化后的数据作为训练数据
            X_train = X_train_scaled
            X_val = X_val_scaled
            X_test = X_test_scaled
            
            # 保存特征均值和标准差，用于后续分析和预测
            # 仅保存必要的统计信息，减少存储
            scaler_stats = {
                'center_': scaler.center_,
                'scale_': scaler.scale_
            }
            joblib.dump(scaler_stats, f"{window_model_dir}/feature_scaler.pkl")
        
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
            # 检查n_bayesian_iter是否满足最小要求
            if n_bayesian_iter < 7:
                print(f"警告: n_bayesian_iter={n_bayesian_iter} 小于最小需求(7)，使用默认参数")
                raise ValueError(f"Expected `n_calls` >= 7, got {n_bayesian_iter}")
    
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
            'best_params': best_params,  # 只保存参数，不保存完整模型对象
            'features_used': window_features,
            'metrics': window_metric,
            'feature_importance': best_model.get_score(importance_type='gain'),
            'model_path': f"{window_model_dir}/xgb_model.json"  # 保存模型路径而非模型对象
        })
        
        # 打印当前窗口评估结果
        print(f"测试集评估指标:")
        print(f"MSE: {test_mse:.10f}, MAE: {test_mae:.10f}")
        print(f"R2: {test_r2:.6f}, IC: {test_ic:.6f}")
        print(f"方向准确率: {test_direction_accuracy:.6f}")
        print(f"盈亏比: {test_profit_loss_ratio:.6f}")
        print(f"符号正确时的平均收益: {avg_return_correct:.10f}")
        
        # 释放内存
        del X_train, X_val, X_test, y_train, y_val, y_test, dtrain, dval, dtest
        # 显式删除模型对象，避免在window_results中保留引用
        del best_model
        import gc
        gc.collect()  # 强制垃圾回收
        
        # 从每个窗口结果中提取特征重要性信息
        for feature, importance in window_results[-1]['feature_importance'].items():
            feature_importances.append({
                'feature': feature,
                'importance': importance,
                'window': window_results[-1]['window']
            })
    
    # 创建整体结果汇总前，检查特征重要性数据
    print(f"收集了 {len(feature_importances)} 条特征重要性记录，来自 {len(window_results)} 个窗口")
    
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
    
    # 将valid_mask转换为numpy数组，用于后续处理
    valid_mask_array = np.array(valid_mask)
    
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
    detailed_metrics_df = metrics_df.copy()
    
    # 添加窗口特定信息
    for i, result in enumerate(window_results):
        if i < len(detailed_metrics_df):
            # 添加使用的特征数量
            detailed_metrics_df.loc[i, 'num_features_used'] = len(result['features_used'])
            
            # 添加模型参数
            for param, value in result['best_params'].items():
                detailed_metrics_df.loc[i, f'param_{param}'] = value
            
            # 添加迭代次数不再从模型对象中获取，因为模型对象已被释放以节省内存
        
    # 保存详细窗口指标
    detailed_metrics_df.to_csv(f"{output_dir}/detailed_window_metrics.csv", index=False)
    
    # 分析参数和性能指标的相关性
    try:
        if len(detailed_metrics_df) > 10:  # 确保有足够的数据进行分析
            # 提取参数列（以'param_'开头的列）
            param_cols = [col for col in detailed_metrics_df.columns if col.startswith('param_')]
            
            if param_cols:  # 确保有参数列
                # 性能指标列
                metric_cols = ['mse', 'r2', 'ic', 'direction_accuracy', 'profit_loss_ratio']
                
                # 创建相关性结果DataFrame
                corr_results = []
                
                # 批量计算Spearman相关系数，减少内存使用
                batch_size = min(50, len(detailed_metrics_df))
                for param_col in param_cols:
                    for metric_col in metric_cols:
                        # 检查参数列是否包含足够的不同值
                        unique_vals = detailed_metrics_df[param_col].nunique()
                        if unique_vals <= 1:  # 跳过只有一个值的参数
                            continue
                            
                        # 确保两列都有有效数值
                        valid_data = detailed_metrics_df[[param_col, metric_col]].dropna()
                        if len(valid_data) < 10:  # 至少需要10个有效样本
                            continue
                            
                        try:
                            # 随机抽样计算相关性以减少内存使用
                            if len(valid_data) > batch_size:
                                sample_data = valid_data.sample(batch_size, random_state=42)
                            else:
                                sample_data = valid_data
                                
                            # 计算Spearman相关系数
                            corr, p_value = stats.spearmanr(
                                sample_data[param_col].values, 
                                sample_data[metric_col].values,
                                nan_policy='omit'
                            )
                            
                            # 只保留显著的相关性（p值<0.1）
                            if not np.isnan(corr) and p_value < 0.1:
                                corr_results.append({
                                    'Parameter': param_col.replace('param_', ''),
                                    'Metric': metric_col,
                                    'Correlation': corr,
                                    'P_Value': p_value,
                                    'Significant': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else '*')
                                })
                        except Exception as e:
                            print(f"计算 {param_col} 和 {metric_col} 相关性时出错: {str(e)}")
                            continue
                
                # 创建相关性DataFrame
                if corr_results:
                    corr_df = pd.DataFrame(corr_results)
                    corr_df = corr_df.sort_values('P_Value')  # 按p值排序
                    
                    # 保存相关性结果
                    corr_df.to_csv(f"{output_dir}/param_performance_correlation.csv", index=False)
                    print(f"参数-性能相关性分析已保存至: {output_dir}/param_performance_correlation.csv")
                    
                    # 绘制热图 - 只绘制相关性强的部分
                    if len(corr_df) > 3:  # 至少需要3个相关性结果才绘制热图
                        try:
                            # 用于绘制热图的数据准备
                            sig_corr_df = corr_df[corr_df['P_Value'] < 0.05].copy()  # 只用显著的相关性
                            if len(sig_corr_df) >= 3:  # 确保有足够的数据绘制热图
                                pivot_df = sig_corr_df.pivot(
                                    index='Parameter', 
                                    columns='Metric', 
                                    values='Correlation'
                                )
                                
                                # 绘制热图
                                plt.figure(figsize=(10, max(4, len(pivot_df) * 0.4)))  # 动态调整图形高度
                                sns.heatmap(
                                    pivot_df, 
                                    annot=True, 
                                    cmap='coolwarm', 
                                    center=0,
                                    linewidths=0.5,
                                    fmt='.2f',
                                    cbar_kws={'label': 'Spearman Correlation Coefficient'}
                                )
                                plt.title('Significant Correlation between Parameters and Performance')
                                plt.tight_layout()
                                plt.savefig(f"{output_dir}/param_performance_heatmap.png", dpi=100)
                                plt.close()
                        except Exception as e:
                            print(f"绘制参数相关性热图时出错: {str(e)}")
            else:
                print("没有找到参数列进行相关性分析")
        else:
            print("窗口数量不足，跳过参数-性能相关性分析")
    except Exception as e:
        print(f"执行参数-性能相关性分析时出错: {str(e)}")
        traceback.print_exc()
    
    # 创建分类表格 - 按性能分组分析窗口
    try:
        if len(detailed_metrics_df) >= 15:  # 确保有足够的窗口进行分组分析
            print("创建窗口性能分组分析...")
            
            # 使用IC值对窗口进行分组
            detailed_metrics_df['IC_Group'] = pd.qcut(
                detailed_metrics_df['ic'], 
                q=3, 
                labels=['Low IC', 'Medium IC', 'High IC'],
                duplicates='drop'  # 处理重复边界值
            )
            
            # 如果分组失败(例如，太多重复值)，使用自定义分组
            if 'IC_Group' not in detailed_metrics_df.columns or detailed_metrics_df['IC_Group'].isna().all():
                ic_median = detailed_metrics_df['ic'].median()
                detailed_metrics_df['IC_Group'] = detailed_metrics_df['ic'].apply(
                    lambda x: 'High IC' if x > ic_median * 1.2 else ('Low IC' if x < ic_median * 0.8 else 'Medium IC')
                )
            
            # 对每个组计算平均指标
            group_stats = []
            for group_name, group_data in detailed_metrics_df.groupby('IC_Group'):
                if len(group_data) < 3:  # 跳过样本过少的组
                    continue
                    
                # 计算基本指标的平均值和标准差
                metric_columns = ['mse', 'r2', 'ic', 'direction_accuracy', 'profit_loss_ratio']
                stats_dict = {'Group': group_name, 'Windows': len(group_data)}
                
                for metric in metric_columns:
                    if metric in group_data.columns:
                        valid_values = group_data[metric].dropna()
                        if len(valid_values) >= 3:
                            stats_dict[f'Avg_{metric}'] = valid_values.mean()
                            stats_dict[f'Std_{metric}'] = valid_values.std()
                
                # 提取参数的平均值
                param_columns = [col for col in group_data.columns if col.startswith('param_')]
                for param in param_columns:
                    valid_values = group_data[param].dropna()
                    if len(valid_values) >= 3 and valid_values.nunique() > 1:
                        param_name = param.replace('param_', '')
                        stats_dict[f'Avg_{param_name}'] = valid_values.mean()
                
                group_stats.append(stats_dict)
            
            # 创建分组统计DataFrame
            if group_stats:
                group_df = pd.DataFrame(group_stats)
                
                # 保存分组分析结果
                group_df.to_csv(f"{output_dir}/performance_group_analysis.csv", index=False)
                print(f"窗口性能分组分析已保存至: {output_dir}/performance_group_analysis.csv")
            else:
                print("分组分析未生成有效结果，可能是因为组内数据不足")
        else:
            print("窗口数量不足，跳过窗口性能分组分析")
    except Exception as e:
        print(f"执行窗口性能分组分析时出错: {str(e)}")
        traceback.print_exc()
    
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
    try:
        valid_mask = ~(all_predictions.isna() | valid_returns.isna())
        if np.sum(valid_mask) > 0:
            corr = pearsonr(all_predictions[valid_mask].values, valid_returns.values)[0]
            print(f"  - 皮尔逊相关系数: {corr:.6f}")
    except Exception as e:
        print(f"  - 计算相关系数时出错: {str(e)}")
    
    # 计算并可视化平均特征重要性
    try:
        if feature_importances and len(feature_importances) > 0:
            # 转换为DataFrame
            importance_df = pd.DataFrame(feature_importances)
            
            # 计算每个特征的平均重要性和使用频率
            avg_importance = importance_df.groupby('feature').agg({
                'importance': ['mean', 'std', 'count']
            }).reset_index()
            
            # 整理列名
            avg_importance.columns = ['feature', 'importance_mean', 'importance_std', 'usage_count']
            
            # 计算使用百分比
            total_windows = len(feature_importances) / len(importance_df['feature'].unique())
            avg_importance['usage_pct'] = avg_importance['usage_count'] / total_windows * 100
            
            # 按平均重要性排序
            avg_importance = avg_importance.sort_values('importance_mean', ascending=False)
            
            # 保存到CSV
            avg_importance.to_csv(f"{output_dir}/avg_feature_importance.csv", index=False)
            print(f"平均特征重要性已保存至: {output_dir}/avg_feature_importance.csv")
            
            # 可视化顶部特征（最多10个以减少内存使用）
            n_top_features = min(10, len(avg_importance))
            if n_top_features > 1:  # 确保至少有2个特征才创建图表
                top_features = avg_importance.head(n_top_features).copy()
                
                # 绘制重要性条形图
                plt.figure(figsize=(8, 5))  # 减小图表尺寸
                
                # 创建条形图
                bars = plt.barh(
                    top_features['feature'],
                    top_features['importance_mean'],
                    xerr=top_features['importance_std'],
                    alpha=0.6,
                    color='darkblue',
                    error_kw={'ecolor': 'black', 'elinewidth': 1, 'capsize': 3}
                )
                
                # 添加使用频率标签
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_width() + bar.get_xerr() * 1.05,
                        bar.get_y() + bar.get_height()/2,
                        f"{top_features['usage_pct'].iloc[i]:.1f}%",
                        va='center',
                        fontsize=8
                    )
                
                plt.xlabel('Average Importance', fontsize=10)
                plt.title('Top Feature Importance and Usage Frequency', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/avg_feature_importance.png", dpi=100)
                plt.close()  # 确保释放内存
        else:
            print("没有特征重要性数据可用于可视化")
    except Exception as e:
        print(f"生成特征重要性可视化时出错: {str(e)}")
        traceback.print_exc()
    
    # 可视化每个窗口的IC值和方向准确率
    try:
        if len(metrics_df) >= 5:  # 确保有足够的窗口数据才生成图表
            print("Generating IC and direction accuracy visualization...")
            
            # 创建单个图表展示IC和方向准确率，减少内存消耗
            plt.figure(figsize=(10, 6))
            
            # 绘制IC值
            ax1 = plt.subplot(111)
            ax1.plot(range(len(metrics_df)), metrics_df['ic'], 'o-', color='royalblue', alpha=0.7, label='IC')
            ax1.axhline(y=metrics_df['ic'].mean(), color='royalblue', linestyle='--', 
                        label=f'Avg IC: {metrics_df["ic"].mean():.4f}')
            ax1.set_ylabel('Information Coefficient (IC)', color='royalblue')
            ax1.tick_params(axis='y', labelcolor='royalblue')
            ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            
            # 创建共享x轴的第二个y轴 - 方向准确率
            ax2 = ax1.twinx()
            ax2.plot(range(len(metrics_df)), metrics_df['direction_accuracy'], 'o-', 
                     color='darkorange', alpha=0.7, label='Direction Accuracy')
            ax2.axhline(y=metrics_df['direction_accuracy'].mean(), color='darkorange', linestyle='--',
                        label=f'Avg Direction Accuracy: {metrics_df["direction_accuracy"].mean():.4f}')
            ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random Level (0.5)')
            ax2.set_ylabel('Direction Accuracy', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            
            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            
            plt.title('IC and Direction Accuracy by Window')
            plt.xlabel('Window Index')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/window_ic_and_direction.png", dpi=100)
            plt.close()  # 确保释放内存
        else:
            print("窗口数量不足，跳过IC和方向准确率可视化")
    except Exception as e:
        print(f"生成IC和方向准确率可视化时出错: {str(e)}")
        traceback.print_exc()
    
    # 绘制预测值与实际值的散点图
    try:
        print("Generating predicted vs actual values scatter plot...")
        
        # 确保有足够的有效预测数据
        valid_preds_mask = ~np.isnan(all_predictions) & ~np.isinf(all_predictions)
        valid_returns_mask = ~np.isnan(valid_returns) & ~np.isinf(valid_returns)
        
        if valid_mask_array.sum() > 0 and valid_preds_mask.sum() > 0 and valid_returns_mask.sum() > 10:
            # 获取有效数据索引
            valid_indices = np.where(valid_mask_array)[0]
            
            # 限制可视化点数，减少内存使用
            max_points = min(3000, len(valid_indices))
            
            # 随机采样以减少点数
            if len(valid_indices) > max_points:
                np.random.seed(42)  # 固定随机种子以便复现
                sampled_indices = np.random.choice(valid_indices, max_points, replace=False)
                sampled_mask = np.zeros_like(valid_mask_array, dtype=bool)
                sampled_mask[sampled_indices] = True
                
                # 确定最终用于绘图的数据
                plot_mask = sampled_mask & valid_preds_mask
                plot_predictions = all_predictions[plot_mask]
                
                # 获取对应的实际收益率
                valid_indices_in_returns = np.where(valid_mask_array[plot_mask])[0]
                if len(valid_indices_in_returns) > 0:
                    plot_returns = valid_returns.iloc[valid_indices_in_returns]
                else:
                    print("采样后没有有效数据点，跳过散点图生成")
                    return
            else:
                # 数据点较少时直接使用全部有效点
                plot_mask = valid_mask_array & valid_preds_mask
                plot_predictions = all_predictions[plot_mask]
                plot_returns = valid_returns
            
            # 计算预测值和实际值的范围
            p_min, p_max = np.percentile(plot_predictions, [1, 99])  # 使用百分位数减少离群值影响
            r_min, r_max = np.percentile(plot_returns, [1, 99])
            
            # 创建散点图
            plt.figure(figsize=(8, 8))
            
            # 添加对角线（理想情况）
            min_val = min(p_min, r_min)
            max_val = max(p_max, r_max)
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Ideal Prediction Line')
            
            # 绘制散点图
            plt.scatter(plot_predictions, plot_returns, alpha=0.5, s=10, color='darkblue')
            
            # 添加趋势线
            try:
                z = np.polyfit(plot_predictions, plot_returns, 1)
                p = np.poly1d(z)
                plt.plot(
                    [p_min, p_max], 
                    [p(p_min), p(p_max)], 
                    'g-', 
                    label=f'Fitted Line (Slope={z[0]:.4f})'
                )
            except Exception as e:
                print(f"绘制趋势线失败: {str(e)}")
            
            # 添加统计信息
            corr = np.corrcoef(plot_predictions, plot_returns)[0, 1]
            plt.text(
                0.05, 0.95, 
                f"Correlation: {corr:.4f}\nPoints: {len(plot_predictions)}", 
                transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            plt.title('Predicted vs Actual Returns')
            plt.xlabel('Predicted Value')
            plt.ylabel('Actual Return')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # 保存图片并释放内存
            plt.savefig(f"{output_dir}/predictions_vs_returns.png", dpi=100)
            plt.close()
        else:
            print("有效预测数据不足，跳过散点图生成")
    except Exception as e:
        print(f"生成预测值与实际值散点图时出错: {str(e)}")
        traceback.print_exc()
    
    # 绘制R2和MSE趋势图（可选）
    # 只在窗口数量足够多时生成，避免过多图形
    if len(metrics_df) >= 5:  # 至少有5个窗口才绘制趋势图
        print("Generating R² and MSE trend plot...")
        plt.figure(figsize=(10, 6))
        
        ax1 = plt.subplot(111)
        ax1.plot(range(len(metrics_df)), metrics_df['r2'], 'o-', alpha=0.7, color='purple', label='R²')
        ax1.set_ylabel('R²', color='purple')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # 创建共享x轴的第二个y轴
        ax2 = ax1.twinx()
        ax2.semilogy(range(len(metrics_df)), metrics_df['mse'], 'o-', alpha=0.7, color='teal', label='MSE')
        ax2.set_ylabel('MSE (log)', color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('R² and MSE Trends by Window')
        plt.xlabel('Window Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/window_r2_and_mse.png", dpi=100)
        plt.close()  # 确保释放内存
    
    # 绘制分位数分析图
    try:
        print("Generating quantile return analysis plot...")
        
        # 创建预测与实际收益的数据框
        pred_df = pd.DataFrame({
            'pred': all_predictions[valid_mask],
            'actual': valid_returns
        }).dropna()
        
        # 检查是否有足够的数据进行分析
        if len(pred_df) >= 100:
            # 减少数据量，抽样处理
            if len(pred_df) > 10000:
                pred_df = pred_df.sample(10000, random_state=42)
            
            plt.figure(figsize=(10, 6))
            
            # 将预测分成10个分位数
            pred_df['quantile'] = pd.qcut(pred_df['pred'], 10, labels=False)
            
            # 计算每个分位数的平均实际收益率
            quantile_returns = pred_df.groupby('quantile')['actual'].mean()
            quantile_counts = pred_df.groupby('quantile').size()
            
            # 绘制每个分位数的平均收益率
            bars = plt.bar(range(10), quantile_returns, alpha=0.7)
            
            # 添加样本数量标签
            for i, bar in enumerate(bars):
                plt.text(
                    i,
                    bar.get_height() + (0.0001 if bar.get_height() >= 0 else -0.0004),
                    f"n={quantile_counts.iloc[i]}",
                    ha='center',
                    va='bottom' if bar.get_height() >= 0 else 'top',
                    fontsize=8
                )
            
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xticks(range(10), [f'Q{i+1}' for i in range(10)])
            plt.xlabel('Prediction Quantiles (Low to High)')
            plt.ylabel('Average Actual Return')
            plt.title('Return Analysis by Prediction Quantile')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/quantile_returns.png", dpi=100)  # 降低dpi以减少内存使用
            plt.close()  # 确保释放内存
        else:
            print("有效数据不足，跳过分位数分析图生成")
    except Exception as e:
        print(f"生成预测分位数收益分析图时出错: {str(e)}")
        traceback.print_exc()
    
    # 保存完整预测结果
    prediction_df = pd.DataFrame({
        'datetime': df['DateTime'],
        'tradday': df['TradDay'],
        'instruID': df['InstruID'],
        'actual_return': df[target_col],
        'prediction': all_predictions
    })
    
    # 按批次保存，减少内存使用
    chunk_size = 100000  # 每个批次的行数
    n_chunks = (len(prediction_df) + chunk_size - 1) // chunk_size  # 向上取整
    
    # 创建一个空的CSV文件，写入列名
    prediction_df.iloc[0:0].to_csv(f"{output_dir}/all_predictions.csv", index=False)
    
    # 按批次追加数据
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(prediction_df))
        chunk = prediction_df.iloc[start_idx:end_idx]
        # 追加模式写入CSV
        chunk.to_csv(f"{output_dir}/all_predictions.csv", mode='a', header=False, index=False)
        # 清理临时数据
        del chunk
    
    # 清理预测数据框以释放内存
    del prediction_df
    
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
        'feature_importance': avg_importance,
    }
    
    return results

def plot_feature_importance(importance_df, output_dir):
    """绘制特征重要性图"""
    try:
        plt.figure(figsize=(12, 8))
        # 只使用前20个特征
        top_features = importance_df.head(20)
        
        # 创建条形图
        bars = plt.barh(range(len(top_features)), 
                       top_features['importance_mean'],
                       xerr=top_features['importance_std'],
                       capsize=5)
        
        # 设置y轴标签
        plt.yticks(range(len(top_features)), top_features['feature'])
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}',
                    va='center', ha='left')
        
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, 'avg_feature_importance.png'))
        plt.close()
        
    except Exception as e:
        print(f"生成特征重要性可视化时出错: {str(e)}")
        traceback.print_exc()

def plot_quantile_returns(pred_df, output_dir):
    """绘制分位数收益分析图"""
    try:
        # 使用duplicates参数处理重复值
        pred_df['quantile'] = pd.qcut(pred_df['pred'], 10, labels=False, duplicates='drop')
        
        # 计算每个分位数的平均收益
        quantile_returns = pred_df.groupby('quantile')['actual'].mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(quantile_returns)), quantile_returns.values, 'o-')
        plt.title('Average Returns by Prediction Quantile')
        plt.xlabel('Prediction Quantile')
        plt.ylabel('Average Return')
        plt.grid(True)
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, 'quantile_returns.png'))
        plt.close()
        
    except Exception as e:
        print(f"生成预测分位数收益分析图时出错: {str(e)}")
        traceback.print_exc()

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
