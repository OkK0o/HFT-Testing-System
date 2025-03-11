from factor_compute import run_factor_compute
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from typing import Dict, List, Tuple, Union, Optional

def get_model(method: str) -> Union[Pipeline, xgb.XGBRegressor, lgb.LGBMRegressor]:
    """
    获取模型实例
    
    Args:
        method: 模型方法名称
        
    Returns:
        模型实例
    """
    if method == 'linear':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
    elif method == 'xgboost':
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
    elif method == 'lightgbm':
        return lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
    elif method == 'random_forest':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100))
        ])
    elif method == 'svr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf'))
        ])
    elif method == 'mlp':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(100, 50)))
        ])
    elif method == 'decision_tree':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeRegressor())
        ])
    else:
        raise ValueError(f"Unsupported method: {method}")

def generate_signals(
    df: pd.DataFrame, 
    target_col: str = 'mid_price', 
    method: str = 'linear',
    loss_function = mean_squared_error,
    train_size: float = 0.8,
    data_using_method: str = "expanding",
    feature_cols: Optional[List[str]] = None,
    forward_periods: int = 1,
    min_train_samples: int = 1000
) -> Tuple[pd.DataFrame, Dict]:
    """
    生成交易信号
    
    Args:
        df: 输入数据
        target_col: 目标列名称
        method: 模型方法，可选值为'linear', 'xgboost', 'lightgbm', 'random_forest', 'svr', 'mlp', 'decision_tree'
        loss_function: 损失函数，默认为MSE
        train_size: 训练集比例
        data_using_method: 训练方法，'expanding'或'rolling'
        feature_cols: 特征列名列表，如果为None则使用所有数值列
        forward_periods: 预测未来多少期的收益率
        min_train_samples: 最小训练样本量
        
    Returns:
        signals: 包含预测信号的DataFrame
        metrics: 模型评估指标字典
    """
    result = df.copy()
    
    if feature_cols is None:
        exclude_cols = [target_col, 'DateTime', 'TradDay', 'InstruID', 'UpdateTime']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
    
    result[f'future_return_{forward_periods}'] = result.groupby('InstruID')[target_col].transform(
        lambda x: x.shift(-forward_periods) / x - 1
    )
    
    result = result.dropna(subset=feature_cols + [f'future_return_{forward_periods}'])
    result = result.sort_values('DateTime')
    
    result['predicted_signal'] = np.nan
    metrics = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    unique_dates = result['TradDay'].unique()
    n_dates = len(unique_dates)
    train_dates = int(n_dates * train_size)
    
    if data_using_method == "expanding":
        for i in range(train_dates, n_dates):
            train_data = result[result['TradDay'] <= unique_dates[i-1]]
            if len(train_data) < min_train_samples:
                continue
            test_data = result[result['TradDay'] == unique_dates[i]]
            model = get_model(method)
            X_train = train_data[feature_cols]
            y_train = train_data[f'future_return_{forward_periods}']
            model.fit(X_train, y_train)
            X_test = test_data[feature_cols]
            predictions = model.predict(X_test)
            result.loc[test_data.index, 'predicted_signal'] = predictions
            y_test = test_data[f'future_return_{forward_periods}']
            if len(y_test) > 0:
                metrics['mse'].append(mean_squared_error(y_test, predictions))
                metrics['mae'].append(mean_absolute_error(y_test, predictions))
                metrics['r2'].append(r2_score(y_test, predictions))
                
    elif data_using_method == "rolling":
        window_size = min_train_samples
        for i in range(train_dates, n_dates):
            train_data = result[
                (result['TradDay'] <= unique_dates[i-1]) & 
                (result['TradDay'] > unique_dates[max(0, i-window_size)])
            ]
            
            if len(train_data) < min_train_samples:
                continue
            test_data = result[result['TradDay'] == unique_dates[i]]
            model = get_model(method)
            X_train = train_data[feature_cols]
            y_train = train_data[f'future_return_{forward_periods}']
            model.fit(X_train, y_train)
            X_test = test_data[feature_cols]
            predictions = model.predict(X_test)
            result.loc[test_data.index, 'predicted_signal'] = predictions
            y_test = test_data[f'future_return_{forward_periods}']
            if len(y_test) > 0:
                metrics['mse'].append(mean_squared_error(y_test, predictions))
                metrics['mae'].append(mean_absolute_error(y_test, predictions))
                metrics['r2'].append(r2_score(y_test, predictions))
    for metric in metrics:
        if metrics[metric]:
            metrics[metric] = np.mean(metrics[metric])
        else:
            metrics[metric] = np.nan
    
    return result, metrics
def plot_signals(df: pd.DataFrame, 
                signal_col: str = 'predicted_signal',
                price_col: str = 'mid_price',
                title: str = 'Predicted Signals vs Price') -> None:
    """
    绘制信号和价格的对比图
    
    Args:
        df: 包含信号和价格的DataFrame
        signal_col: 信号列名
        price_col: 价格列名
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    ax1.plot(df['DateTime'], df[price_col], label='Price')
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(df['DateTime'], df[signal_col], label='Signal', color='orange')
    ax2.set_ylabel('Signal')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names: List[str], title: str = 'Feature Importance') -> None:
    """
    绘制特征重要性图
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        title: 图表标题
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("This model doesn't support feature importance visualization")
        return
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_feather("data.feather")
    result, metrics = generate_signals(
        df=df,
        target_col='mid_price',
        method='xgboost',
        data_using_method='expanding',
        forward_periods=5
    )
    
    print("\n模型评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    plot_signals(result)
    