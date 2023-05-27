# https://towardsdatascience.com/how-to-combine-the-forecasts-of-an-ensemble-11022e5cac25
# https://github.com/vcerqueira/blog/blob/main/src/ensembles/windowing.py

# 静态集成：对每个模型在每个时刻分配相同的权重, 0.3*modelA_result + 0.7*modelB_result
# 动态集成：constant weights fail to adapt to changes in the time series
# 动态集成Ensembles with dynamic weights estimating which models are stronger in a given instant
'''
Windowing: the weights are computed based on model performance in a window of past recent data. 
For example, you can compute the rolling squared error. Then, at each instant, you get the weights by normalizing 
the error scores.

Regret minimization: Some methods attempt to minimize a metric called regret. 
Examples include the exponentially weighted average, the polynomially weighted average, or the fixed share aggregation.

Meta-learning: Other techniques learn and predict the weights of each model for a given instant.
'''

import pandas as pd

# methods and validation split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNetCV

# time series example -- GPL-3 license
from pmdarima.datasets import load_taylor

# src module available here: https://github.com/vcerqueira/blog
from src.tde import time_delay_embedding

# loading the data
series = load_taylor(as_series=True)

# train test split
train, test = train_test_split(series, test_size=0.1, shuffle=False)

# ts for supervised learning
train_df = time_delay_embedding(train, n_lags=10, horizon=1).dropna()
test_df = time_delay_embedding(test, n_lags=10, horizon=1).dropna()

# creating the predictors and target variables
# the goal is to forecast the next observation of energy demand
X_train, y_train = train_df.drop('Series(t+1)', axis=1), train_df['Series(t+1)']
X_test, y_test = test_df.drop('Series(t+1)', axis=1), test_df['Series(t+1)']

# defining the five models composing the ensemble
models = {
    'RF': RandomForestRegressor(),
    'KNN': KNeighborsRegressor(),
    'LASSO': Lasso(),
    'EN': ElasticNetCV(),
    'Ridge': Ridge(),
}

# training and getting predictions
train_forecasts, test_forecasts = {}, {}
for k in models:
    models[k].fit(X_train, y_train)
    train_forecasts[k] = models[k].predict(X_train)
    test_forecasts[k] = models[k].predict(X_test)

# predictions as pandas dataframe
ts_forecasts_df = pd.DataFrame(test_forecasts)
tr_forecasts_df = pd.DataFrame(train_forecasts)


# ===============动态权重估计及集成=============
# src module available at https://github.com/vcerqueira/blog
# windowing 
from src.ensembles.windowing import WindowLoss
# arbitrating (a meta-learning strategy)
from src.ensembles.ade import Arbitrating

# combining training and testing predictions
forecasts_df = pd.concat([tr_forecasts_df, ts_forecasts_df], axis=0).reset_index(drop=True)
# combining training and testing observations
actual = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

# setting up windowloss dynamic combinatio rule
windowing = WindowLoss()
window_weights = windowing.get_weights(forecasts_df, actual)
window_weights = window_weights.tail(X_test.shape[0]).reset_index(drop=True)

# setting up arbitrating dynamic combinatio rule
arbitrating = Arbitrating()
arbitrating.fit(tr_forecasts_df, y_train, X_train)
arb_weights = arbitrating.get_weights(X_test)
arb_weights = arb_weights.tail(X_test.shape[0])

# weighting the ensemble dynamically
windowing_fh = (window_weights.values * ts_forecasts_df.values).sum(axis=1)
arbitrating_fh = (arb_weights.values * ts_forecasts_df.values).sum(axis=1)

# ======================静态集成-简单平均================
# combining the models with static and equal weights (average)
static_average = ts_forecasts_df.mean(axis=1).values