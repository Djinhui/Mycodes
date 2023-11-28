#!/usr/bin/env python
# coding: utf-8

# combine nbeats_darts.py and nhits_darts.py

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries



df = pd.read_csv('daily_traffic.csv')
df.head()


series = TimeSeries.from_dataframe(df, time_col='date_time')
series.plot()


from darts.utils.statistics import check_seasonality

# is_seasonal, period = check_seasonality(series, max_lag=1000, alpha=0.05) # 不指定m
is_daily_seasonal, daily_period = check_seasonality(series, m=24, max_lag=400, alpha=0.05)
is_weekly_seasonal, weekly_period = check_seasonality(series, m=168, max_lag=400, alpha=0.05)

print(f'Daily seasonality: {is_daily_seasonal} - period = {daily_period}')
print(f'Weekly seasonality: {is_weekly_seasonal} - period = {weekly_period}')



train, test = series[:-120], series[-120:]

train.plot(label='train')
test.plot(label='test')


from darts.models.forecasting.baselines import NaiveSeasonal

naive_seasonal = NaiveSeasonal(K=168)
naive_seasonal.fit(train)

pred_naive = naive_seasonal.predict(120)

test.plot(label='test')
pred_naive.plot(label='Baseline')


from darts.metrics import mae

naive_mae = mae(test, pred_naive)
print(naive_mae)


from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)

nbeats = NBEATSModel(
    input_chunk_length=168, 
    output_chunk_length=24,
    generic_architecture=False,
    random_state=42)

'''
    num_stacks = 3,
    num_blocks=3,
    num_layers=4,
    layer_widths=32,
'''

nbeats.fit(
    scaled_train,
    epochs=50
)


scaled_pred_nbeats = nbeats.predict(n=120)
pred_nbeats = train_scaler.inverse_transform(scaled_pred_nbeats)
mae_nbeats = mae(test, pred_nbeats)
print(mae_nbeats)


scaled_pred_nbeats = nbeats.predict(n=120, series=scaled_train[-168:])
pred_nbeats = train_scaler.inverse_transform(scaled_pred_nbeats)
mae_nbeats = mae(test, pred_nbeats)
print(mae_nbeats)

# nbeats.historical_forecasts()
nbeats = NBEATSModel(
    input_chunk_length=168, 
    output_chunk_length=24,
    generic_architecture=True,
    random_state=42)

'''
    num_stacks = 3,
    num_blocks=3,
    num_layers=4,
    layer_widths=32,
'''
nbeats.fit(
    scaled_train,
    epochs=50
)

scaled_pred_nbeats = nbeats.predict(n=120)
pred_nbeats_generic = train_scaler.inverse_transform(scaled_pred_nbeats)
mae_nbeats_generic = mae(test, pred_nbeats_generic)
print(mae_nbeats_generic)



scaled_pred_nbeats = nbeats.predict(n=120, series=scaled_train[-168:])
pred_nbeats_generic = train_scaler.inverse_transform(scaled_pred_nbeats)
mae_nbeats_generic = mae(test, pred_nbeats_generic)
print(mae_nbeats_generic)



from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

cov = concatenate(
    [dt_attr(series.time_index, 'day', dtype=np.float32), dt_attr(series.time_index, 'week', dtype=np.float32)],
    axis='component'
)

cov_scaler = Scaler()
scaled_cov = cov_scaler.fit_transform(cov)
train_scaled_cov, test_scaled_cov = scaled_cov[:-120], scaled_cov[-120:]
scaled_cov.plot()

nbeats_cov = NBEATSModel(
    input_chunk_length=168, 
    output_chunk_length=24,
    generic_architecture=False,
    random_state=42)

nbeats_cov.fit(
    scaled_train,
    past_covariates=scaled_cov,
    epochs=50
)


scaled_pred_nbeats_cov = nbeats_cov.predict(past_covariates=scaled_cov, n=120)
pred_nbeats_cov = train_scaler.inverse_transform(scaled_pred_nbeats_cov)
mae_nbeats_cov = mae(test, pred_nbeats_cov)
print(mae_nbeats_cov)


nbeats_cov = NBEATSModel(
    input_chunk_length=168, 
    output_chunk_length=24,
    generic_architecture=True,
    random_state=42)

'''
    num_stacks = 3,
    num_blocks=3,
    num_layers=4,
    layer_widths=32,
'''

nbeats_cov.fit(
    scaled_train,
    past_covariates=scaled_cov,
    epochs=50
)


scaled_pred_nbeats_cov = nbeats_cov.predict(past_covariates=scaled_cov, n=120)
pred_nbeats_cov_generic = train_scaler.inverse_transform(scaled_pred_nbeats_cov)
mae_nbeats_cov_generic = mae(test, pred_nbeats_cov_generic)
print(mae_nbeats_cov_generic)


from darts.models import NHiTSModel

'''
N-HiTS is similar to N-BEATS (implemented in :class:`NBEATSModel`),
but attempts to provide better performance at lower computational cost by introducing
multi-rate sampling of the inputs and multi-scale interpolation of the outputs.
'''

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
print(mae_nhits)



scaled_pred_nhits2 = nhits.predict(n=120,series=scaled_train[-168:])
pred_nhits2 = train_scaler.inverse_transform(scaled_pred_nhits2)
mae_nhits2 = mae(test, pred_nhits2)
print(mae_nhits2)


test.plot(label='test')
pred_naive.plot(label='Baseline')
pred_nbeats.plot(label='N-BEATS')
pred_nbeats_generic.plot(label='N-BEATS-Generic')
pred_nbeats_cov.plot(label='N-BEATS-COV')
pred_nbeats_cov_generic.plot(label='N-BEATS-Generic-COV')
pred_nhits.plot(label='N-HiTS')


fig, ax = plt.subplots()
x = ['Baseline', 'N-Beats', 'N-BEATS-Generic', 'N-BEATS-COV', 'N-BEATS-Generic-COV', 'N-HiTS']
y = [naive_mae, mae_nbeats, mae_nbeats_generic, mae_nbeats_cov, mae_nbeats_cov_generic, mae_nhits]
ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAE')
# ax.set_ylim(0, 350)
ax.grid(False)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 10, s=str(round(value,0)), ha='center')
plt.tight_layout()

fig, ax = plt.subplots()
x = ['Baseline', 'N-Beats', 'N-BEATS-Generic',  'N-BEATS-Generic-COV', 'N-HiTS']
y = [naive_mae, mae_nbeats, mae_nbeats_generic, mae_nbeats_cov_generic, mae_nhits]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAE')
# ax.set_ylim(0, 350)
ax.grid(False)
for index, value in enumerate(y):
    plt.text(x=index, y=value + 10, s=str(round(value,0)), ha='center')
plt.tight_layout()






from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood

model = TCNModel(
    input_chunk_length=168,
    output_chunk_length=120,
    random_state=42,
    likelihood=LaplaceLikelihood(), # 概率预测
)

model.fit(scaled_train, epochs=400, verbose=True)
pred = model.predict(n=120, num_samples=500) #  ·num_samples· Monte Carlo samples
pred = train_scaler.inverse_transform(pred)