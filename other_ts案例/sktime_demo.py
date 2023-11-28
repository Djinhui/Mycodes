# BATS-TBATS《Forecasting Time Series With Complex Seasonal Patterns Using Exponential Smoothing》

# Sktime :时序分类-聚类-回归-预测-异常检测

# Forecasting  


# imports necessary for this chapter
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series

# data loading for illustration (see section 1 for explanation)
y = load_airline() # pd.Series with time index
y_train, y_test = temporal_train_test_split(y, test_size=36)
fh = ForecastingHorizon(y_test.index, is_relative=False)


# ============================指数平滑
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS

forecaster = ExponentialSmoothing(trend='add', seasonal='additive', sp=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=['train', 'test', 'pred'])
mean_absolute_percentage_error(y_test, y_pred, symmetric=False) # MAPE OR sMAPE 0.05114163

forecaster = AutoETS(auto=True, sp=12, n_jobs=-1)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
mean_absolute_percentage_error(y_test, y_pred, symmetric=False) # 0.06186318


# ==================Theta
from sktime.forecasting.theta import ThetaForecaster
forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

# ==================Arima 
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.arima import AutoARIMA

forecaster = ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh) # 0.0435674488

forecaster = AutoARIMA(sp=12, suppress_warnings=True)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh) # 0.04148971438


# =================TBATS
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS

forecaster = BATS(sp=12, use_trend=True, use_box_cox=False)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh) # 0.081855589

forecaster = TBATS(sp=12, use_trend=True, use_box_cox=False)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh) # 0.080240908

'''
# 2. pip install tbats
from tbats import BATS, TBATS
from tbats import TBATS
import numpy as np

np.random.seed(2342)
t = np.array(range(0, 160))
y = 5 * np.sin(t * 2 * np.pi / 7) + 2 * np.cos(t * 2 * np.pi / 30.5) + \
    ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10

# Create estimator
estimator = TBATS(seasonal_periods=[14, 30.5])

# Fit model
fitted_model = estimator.fit(y)

# Forecast 14 steps ahead
y_forecasted = fitted_model.forecast(steps=14)

# Summarize fitted model
print(fitted_model.summary())
'''

# ==============prophet
from sktime.forecasting.fbprophet import Prophet
# Convert index to pd.DatetimeIndex
z = y.copy()
z = z.to_timestamp(freq="M")
z_train, z_test = temporal_train_test_split(z, test_size=36)

forecaster = Prophet(
    seasonality_mode="multiplicative",
    n_changepoints=int(len(y_train) / 12),
    add_country_holidays={"country_name": "Germany"},
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)

forecaster.fit(z_train)
y_pred = forecaster.predict(fh.to_relative(cutoff=y_train.index[-1]))
y_pred.index = y_test.index
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
mean_absolute_percentage_error(y_test, y_pred, symmetric=False) # 0.072768629

