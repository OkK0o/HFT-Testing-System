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

def generate_signals(df: pd.DataFrame, target_col: str = 'LastPrice', method: str = 'linear', loss_function = mean_squared_error, train_size: float = 0.8):
    """
    生成交易信号
    
    Args:
        df: 输入数据
        target_col: 目标列名称
        method: 模型方法，可选值为'linear', 'xgboost', 'lightgbm', 'random_forest', 'svr', 'mlp', 'decision_tree'
        loss function: 损失函数，传入方法，可选方法为'mse', 'mae', 'r2', 同时可以传入自定义方法
        train_size: 训练集比例
    Returns:
        signals: 交易信号
    """
    # 数据预处理
    df = df.copy()

    
    