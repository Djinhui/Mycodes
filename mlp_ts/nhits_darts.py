import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from darts import TimeSeries

df = pd.read_csv('daily_traffic.csv')
df.head()  # cols:[date_time, traffic_volume], freq='1h'

series = TimeSeries.from_dataframe(df, time_col='date_time')
series.plot()
plt.xlim('9/29/2016 17:00', '10/16/2016 9:00')

# check seasonality
from darts.utils.statistics import check_seasonality

is_daily_seasonal, daily_period = check_seasonality(series, m=24, max_lag=400, alpha=0.05)
is_weekly_seasonal, weekly_period = check_seasonality(series, m=168, max_lag=400, alpha=0.05)

print(f'Daily seasonality: {is_daily_seasonal} - period = {daily_period}')
print(f'Weekly seasonality: {is_weekly_seasonal} - period = {weekly_period}')
'''
Daily seasonality: True - period = 24
Weekly seasonality: True - period = 168
'''

train, test = series[:-120], series[-120:]
train.plot(label='train')
test.plot(label='test')

# Baseline
from darts.models.forecasting.baselines import NaiveSeasonal

naive_seasonal = NaiveSeasonal(K=168)
naive_seasonal.fit(train)
pred_naive = naive_seasonal.predict(120)

test.plot(label='test')
pred_naive.plot(label='Baseline')

from darts.metrics import mae

naive_mae = mae(test, pred_naive)



# NHits model
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler

train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)

nhits = NHiTSModel(
    input_chunk_length=168, 
    output_chunk_length=120,
    random_state=42)

nhits.fit(
    scaled_train,
    epochs=50)

scaled_pred_nhits = nhits.predict(n=120)
pred_nhits = train_scaler.inverse_transform(scaled_pred_nhits)
mae_nhits = mae(test, pred_nhits)

test.plot(label='Actual')
pred_nhits.plot(label='N-HiTS')


# compare  see nbeats_darts.py
fig, ax = plt.subplots()

x = ['Baseline', 'N-Beats', 'N-BEATS + covariates', 'N-HiTS']
y = [naive_mae, 292, 288, mae_nhits]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAE')
ax.set_ylim(0, 350)
ax.grid(False)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 10, s=str(round(value,0)), ha='center')

plt.tight_layout()