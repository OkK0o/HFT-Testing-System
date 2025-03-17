import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import shap

def xgboost_model(X_train, y_train, X_test, y_test, loss_function = mean_squared_error,min_train_samples = 1000):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# 贝叶斯优化参数空间
param_space = {
    'n_estimators': (50, 200),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

def generate_rolling_signals(df, traget = 'mid_price', loss_function = mean_squared_error, train_size = 0.8, window_size = 4,start_date = None, end_date = None, feature_cols = None, forward_periods = 1, min_train_samples = 1000):
    # 计算mid_price的未来变化
    df['future_change'] = df.groupby('InstruID')[traget].transform(lambda x: x.diff(forward_periods).shift(-forward_periods))
    
    y = df['future_change']
    X = df[feature_cols]
    date = df['TradDay']
    
    # 确保y和X包含日期列
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    date = date.reset_index(drop=True)
    
    if start_date is not None:
        y = y[date >= start_date]
        X = X[date >= start_date]
        date = date[date >= start_date]
    if end_date is not None:
        y = y[date <= end_date]
        X = X[date <= end_date]
        date = date[date <= end_date]
    
    strat = start_date
    time_delta = pd.Timedelta(days=window_size)
    
    models = {}
    time_delta_train = pd.Timedelta(days=3)
    time_delta_test = pd.Timedelta(days=1)
    time_delta_val = pd.Timedelta(days=1)
    
    for i in pd.date_range(start=start_date, end=end_date - time_delta_train - time_delta_test - time_delta_val, freq='D'):
        train_end = i + time_delta_train
        test_end = train_end + time_delta_test
        val_end = test_end + time_delta_val
        
        X_train = X[(date >= i) & (date < train_end)]
        y_train = y[(date >= i) & (date < train_end)]
        X_test = X[(date >= train_end) & (date < test_end)]
        y_test = y[(date >= train_end) & (date < test_end)]
        X_val = X[(date >= test_end) & (date < val_end)]
        y_val = y[(date >= test_end) & (date < val_end)]
        
        if len(X_train) < min_train_samples:
            continue
        
        # 贝叶斯优化
        opt = BayesSearchCV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror'),
            search_spaces=param_space,
            n_iter=30,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        opt.fit(X_train, y_train)
        best_model = opt.best_estimator_
        models[i] = best_model
        
        y_pred_test = best_model.predict(X_test)
        y_pred_val = best_model.predict(X_val)
        
        # SHAP分析
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        
        # 可以在这里计算并存储测试和验证集的评估指标
        # test_metrics = loss_function(y_test, y_pred_test)
        # val_metrics = loss_function(y_val, y_pred_val)
        
    return models

