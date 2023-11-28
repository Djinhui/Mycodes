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

from darts.metrics import mape
naive_mape = mape(test, pred_naive)


# N-Beats without covariates
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)

nbeats = NBEATSModel(input_chunk_length=168, output_chunk_length=24,generic_architecture=True)
nbeats.fit(scaled_train, epochs=50)

scaled_pred_nbeats = nbeats.predict(n=120)
pred_nbeats = train_scaler.inverse_transform(scaled_pred_nbeats)

mape_nbeats = mape(test, pred_nbeats)

# N-Beats with covariates
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

cov = concatenate([
    dt_attr(series.time_index, 'day', dtype=np.float32),
    dt_attr(series.time_index, 'week', dtype=np.float32)
], axis='component')

cov_scaler = Scaler()
scaled_cov = cov_scaler.fit_transform(cov)
train_scaled_cov, test_scaled_cov = scaled_cov[:-120], scaled_cov[-120:]
scaled_cov.plot()


nbeats_cov = NBEATSModel(input_chunk_length=168,
                         output_chunk_length=24,
                         generic_architecture=True)
nbeats_cov.fit(scaled_train, past_covariates=scaled_cov,epochs=50)

scaled_pred_nbeats_cov = nbeats_cov.predict(past_covariates=scaled_cov, n=120)
pred_nbeats_cov = train_scaler.inverse_transform(scaled_pred_nbeats_cov)

mape_nbeats_cov = mape(test, pred_nbeats_cov)

test.plot(label='test')
pred_nbeats.plot(label='N-BEATS')


# Compare - Baseline is best
fig, ax = plt.subplots()

x = ['Baseline', 'N-Beats', 'N-BEATS + covariates']
y = [naive_mape, mape_nbeats, mape_nbeats_cov]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAE')
ax.set_ylim(0, 350)
ax.grid(False)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 10, s=str(round(value,0)), ha='center')

plt.tight_layout()



# =========================================Multiple series forecast============================================
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.dataprocessing.transformers import Scaler

series_air = AirPassengersDataset().load().astype(np.float32)
series_milk = MonthlyMilkDataset().load().astype(np.float32)

# set aside last 36 months of each series as validation set:
train_air, val_air = series_air[:-36], series_air[-36:]
train_milk, val_milk = series_milk[:-36], series_milk[-36:]

scaler = Scaler()
train_air_scaled, train_milk_scaled = scaler.fit_transform([train_air, train_milk])
train_air_scaled.plot()
train_milk_scaled.plot()


model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, random_state=42)
model.fit([train_air_scaled, train_milk_scaled], epochs=50, verbose=True)

# use the series argument of the fit() function to tell the model which series to forecast
pred_air = model.predict(series=train_air_scaled, n=36)
pred_milk = model.predict(series=train_milk_scaled, n=36)

# scale back:
pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])

plt.figure(figsize=(10, 6))
series_air.plot(label="actual (air)")
series_milk.plot(label="actual (milk)")
pred_air.plot(label="forecast (air)")
pred_milk.plot(label="forecast (milk)")


# =====================================Covariates: using external data===========================
'''
Covariates are series that we do not want to forecast, but which can provide helpful additional information to the models. Both the targets and covariates series can be multivariate or univariate.
'''
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

air_covs = concatenate(
    [
        dt_attr(series_air.time_index, "month", dtype=np.float32) / 12,
        (dt_attr(series_air.time_index, "year", dtype=np.float32) - 1948) / 12,
    ],
    axis="component",
)

milk_covs = concatenate(
    [
        dt_attr(series_milk.time_index, "month", dtype=np.float32) / 12,
        (dt_attr(series_milk.time_index, "year", dtype=np.float32) - 1962) / 13,
    ],
    axis="component",
)

air_covs.plot()
plt.title(
    "one multivariate time series of 2 dimensions, containing covariates for the air series:"
)

'''
NBEATSModel supports only `past_covariates`. Therefore, even though our covariates represent calendar information and are known in advance, we will use them as past_covariates with N-BEATS. 
'''

model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, random_state=42)
model.fit(
    [train_air_scaled, train_milk_scaled],
    past_covariates=[air_covs, milk_covs],
    epochs=50,
    verbose=True,
)

# equals to 
# encoders = {"datetime_attribute": {"past": ["month", "year"]}, "transformer": Scaler()}
# model = NBEATSModel(
#     input_chunk_length=24,
#     output_chunk_length=12,
#     add_encoders=encoders,
#     random_state=42,
# )

# model.fit([train_air_scaled, train_milk_scaled], epochs=50, verbose=True)

'''
Even though the covariates time series also contains “future” values of the covariates up to the forecast horizon, 
the model will not consume those future values, because it uses them as past covariates (and not future covariates).
'''
pred_air = model.predict(series=train_air_scaled, past_covariates=air_covs, n=36)
pred_milk = model.predict(series=train_milk_scaled, past_covariates=milk_covs, n=36)

# scale back:
pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])