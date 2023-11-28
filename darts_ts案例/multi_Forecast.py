# ==============================================Support for multivariate series====================================
'''
Some models support multivariate time series. This means that the target (and potential covariates) series provided to the model during fit and predict stage 
can have multiple dimensions. The model will then in turn produce multivariate forecasts.
'''
import darts.utils.timeseries_generation as tg
from darts.models import KalmanForecaster
import matplotlib.pyplot as plt
import numpy as np

series1 = tg.sine_timeseries(value_frequency=0.05, length=100) + 0.1 * tg.gaussian_timeseries(length=100)
series2 = tg.sine_timeseries(value_frequency=0.02, length=100) + 0.2 * tg.gaussian_timeseries(length=100)

multivariate_series = series1.stack(series2) # 2 dimensions/components

# using a KalmanForecaster to forecast a single multivariate series made of 2 components
model = KalmanForecaster(dim_x=4)
model.fit(multivariate_series)
pred = model.predict(n=50, num_samples=100)

plt.figure(figsize=(8,6))
multivariate_series.plot(lw=3)
pred.plot(lw=3, label='forecast')


# =========================================Multiple series forecast============================================
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel

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


'''
model.fit(series=[series1, series2, ...],
          past_covariates=[past_cov1, past_cov2, ...],
          future_covariates=[future_cov1, future_cov2, ...])
future = model.predict(n=36,
                       series=series_to_forecast,
                       past_covariates=past_covariate_series,
                       future_covariates=future_covariate_series)
'''



from darts.models import RNNModel

model = RNNModel()