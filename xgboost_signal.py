import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
import shap
import numpy as np
from scipy import stats

def calculate_ic(predictions, returns):
    """计算IC值"""
    # 去除无效值
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    if not np.any(mask):
        return np.nan
    
    predictions = predictions[mask]
    returns = returns[mask]
    
    if len(predictions) < 30:  # 最小样本量要求
        return np.nan
        
    # 计算Spearman相关系数
    ic = stats.spearmanr(predictions, returns)[0]
    return ic

def xgboost_model(X_train, y_train, X_test, y_test, loss_function = mean_squared_error,min_train_samples = 1000):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

param_space = {
    'n_estimators': (50, 200),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

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

def generate_rolling_signals(df, traget='mid_price', loss_function=mean_squared_error, window_size=4, start_date=None, end_date=None, feature_cols=None, forward_periods=1, min_train_samples=1000):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
    
    df['future_change'] = df.groupby('InstruID')[traget].transform(lambda x: x.diff(forward_periods).shift(-forward_periods))
    
    y = df['future_change']
    X = df[feature_cols]
    date = df['TradDay']
    
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    date = date.reset_index(drop=True)
    
    trading_days = sorted(df['TradDay'].unique())
    trading_days = pd.to_datetime(trading_days)
    
    if start_date is not None:
        start_date = trading_days[trading_days >= start_date][0]
    if end_date is not None:
        end_date = trading_days[trading_days <= end_date][-1]
    
    models = {}
    predictions = pd.Series(index=df.index, dtype=float)  # 存储所有预测结果
    metrics_list = []  # 存储每个时间窗口的评估指标
    
    time_delta_train = pd.Timedelta(days=3)
    time_delta_test = pd.Timedelta(days=1)
    time_delta_val = pd.Timedelta(days=1)
    
    for i in trading_days:
        if i >= end_date - time_delta_train - time_delta_test - time_delta_val:
            break
            
        train_end = trading_days[trading_days > i][2]  
        test_end = trading_days[trading_days > train_end][0]  
        val_end = trading_days[trading_days > test_end][0]  
        
        X_train = X[(date >= i) & (date < train_end)]
        y_train = y[(date >= i) & (date < train_end)]
        X_test = X[(date >= train_end) & (date < test_end)]
        y_test = y[(date >= train_end) & (date < test_end)]
        X_val = X[(date >= test_end) & (date < val_end)]
        y_val = y[(date >= test_end) & (date < val_end)]
        
        if len(X_train) < min_train_samples:
            continue
        
        opt = xgb.XGBRegressor(objective='reg:squarederror', missing=np.nan)
        opt.fit(X_train, y_train)
        best_model = opt
        models[i] = best_model
        
        # 生成预测
        y_pred_test = best_model.predict(X_test)
        y_pred_val = best_model.predict(X_val)
        
        # 存储验证集的预测结果
        val_indices = df.index[(date >= test_end) & (date < val_end)]
        predictions.loc[val_indices] = y_pred_val
        
        # 计算评估指标
        val_mse = mean_squared_error(y_val, y_pred_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        direction_accuracy = np.mean(np.sign(y_val) == np.sign(y_pred_val))
        
        # 计算IC
        ic = calculate_ic(y_pred_val, y_val)
        
        profits = np.sum(np.abs(y_val[y_val * y_pred_val > 0]))
        losses = np.sum(np.abs(y_val[y_val * y_pred_val < 0]))
        profit_loss_ratio = profits / losses if losses != 0 else np.nan
        
        # 存储指标
        metrics = {
            'date': i,
            'mse': val_mse,
            'mae': val_mae,
            'r2': val_r2,
            'direction_accuracy': direction_accuracy,
            'profit_loss_ratio': profit_loss_ratio,
            'ic': ic
        }
        metrics_list.append(metrics)
        
        print(f"\n验证集评估指标 (日期: {i}):")
        print(f"MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        print(f"预测方向正确率: {direction_accuracy:.4f}")
        print(f"盈亏比: {profit_loss_ratio:.4f}")
        print(f"IC: {ic:.4f}")
    
    # 汇总所有时间窗口的指标
    metrics_df = pd.DataFrame(metrics_list)
    
    # 计算整体IC均值
    overall_ic = calculate_ic(predictions.dropna(), df.loc[predictions.dropna().index, 'future_change'])
    
    # 分析信号分布
    signal_stats = analyze_signal_distribution(predictions.dropna(), df.loc[predictions.dropna().index, 'future_change'])
    
    print(f"\n=== 整体评估指标 ===")
    print(f"平均 IC: {metrics_df['ic'].mean():.4f}")
    print(f"整体 IC: {overall_ic:.4f}")
    print(f"平均方向准确率: {metrics_df['direction_accuracy'].mean():.4f}")
    print(f"平均盈亏比: {metrics_df['profit_loss_ratio'].mean():.4f}")
    
    print("\n=== 信号分布统计 ===")
    for key, value in signal_stats.items():
        print(f"{key}: {value:.4f}")
    
    return {
        'models': models,
        'predictions': predictions,
        'metrics': metrics_df,
        'overall_metrics': {
            'mean_ic': metrics_df['ic'].mean(),
            'overall_ic': overall_ic,
            'mean_direction_accuracy': metrics_df['direction_accuracy'].mean(),
            'mean_profit_loss_ratio': metrics_df['profit_loss_ratio'].mean()
        },
        'signal_stats': signal_stats
    }
