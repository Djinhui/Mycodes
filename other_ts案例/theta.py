# <The theta model: A decomposition approach to forecasting>
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')
'''
1. ThetaForecaster in SKtime

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.registry import all_estimators

all_estimators("forecaster", as_dataframe=True)

y = load_airline() # series

y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
print(mean_absolute_percentage_error(y_test, y_pred))
'''
df = sm.datasets.co2.load_pandas().data

fig, ax = plt.subplots()
ax.plot(df['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('Co2')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# missing values
df = df.interpolate()

train = df[:-104]
test = df[-104:] # two years 52weeks per year

def rolling_forecast(df:pd.DataFrame, train_len:int, horizon:int,window:int,method:str):
    total_len = train_len + horizon
    end_idx = train_len

    if method == 'last_season':
        pred_last_season = []
        for i in range(train_len, total_len, window):
            last_season = df[:i].iloc[-window:].values
            pred_last_season.extend(last_season)
        return pred_last_season
    
    elif method == 'theta':
        pred_theta = []
        for i in range(train_len, total_len, window):
            tm = ThetaModel(endog=df[:i], period=52)
            res = tm.fit()
            predictions = res.forecast(window)
            pred_theta.extend(predictions)
        # print(res.summary())
        return pred_theta
    
    elif method == 'tes': # triple exponential smoothing.
        pred_tes = []
        for i in range(train_len, total_len, window):
            tes = ExponentialSmoothing(df[:i],trend='add', seasonal='add', seasonal_periods=52,
                                       initialization_method='estimated').fit()
            predictions = tes.forecast(window)
            pred_tes.extend(predictions)
        return pred_tes
    
    
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 52

pred_last_season = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'last_season')
pred_theta = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'theta')
pred_tes = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'tes')


test = test.copy()
test.loc[:, 'pred_last_season'] = pred_last_season
test.loc[:, 'pred_theta'] = pred_theta
test.loc[:, 'pred_tes'] = pred_tes

fig, ax = plt.subplots()

ax.plot(df['co2'])
ax.plot(test['co2'], 'b-', label='actual')
ax.plot(test['pred_last_season'], 'r:', label='baseline')
ax.plot(test['pred_theta'], 'g-.', label='Theta')
ax.plot(test['pred_tes'], 'k--', label='TES')

ax.set_xlabel('Time')
ax.set_ylabel('CO2 concentration (ppmv)')
ax.axvspan('2000-01-08', '2001-12-29', color='#808080', alpha=0.2)

ax.legend(loc='best')

ax.set_xlim('1998-03-07', '2001-12-29')

fig.autofmt_xdate()
plt.tight_layout()
plt.show()


def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

mape_baseline = mape(test['co2'], test['pred_last_season'])
mape_theta = mape(test['co2'], test['pred_theta'])
mape_tes = mape(test['co2'], test['pred_tes'])

print(mape_baseline, mape_theta, mape_tes) # 0.36 0.28 0.12

fig, ax = plt.subplots()

x = ['Baseline', 'Theta', 'TES']
y = [mape_baseline, mape_theta, mape_tes]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Exponential smoothing models')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 1)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 0.05, s=str(value), ha='center')
    
plt.tight_layout()
plt.show()