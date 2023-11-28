# install with support for torch models
!pip install "gluonts[torch]"

# install with support for mxnet models
!pip install "gluonts[mxnet]"

import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer

dataset = get_dataset('airpassengers')

deepar = DeepAREstimator(prediction_length=12, freq='M', trainer=Trainer(epochs=5))
model = deepar.train(dataset.train)

true_values = to_pandas(list(dataset.test)[0])
true_values.to_timestamp().plot(color='k')

prediction_input = PandasDataset([true_values[:-36], true_values[:-24], true_values[:-12]])
predictions = model.predict(prediction_input)

for color, prediction in zip(['green', 'blue', 'puple'], predictions):
    prediction.plot(color=f'tab:{color}')


from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator as TorchDeepAR

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/"
    "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
    index_col=0,
    parse_dates=True,
)
dataset = PandasDataset(df, target='#Passengers')

training_data, test_gen = split(dataset, offset=-36)
test_data = test_gen.generate_instances(prediction_length=12, windows=3)

model = TorchDeepAR(prediction_length=12, freq='M', trainer_kwargs={'max_epochs':5}).train(training_data)
forecasts = list(model.predict(test_data.input))

plt.plot(df["1954":], color="black")
for forecast in forecasts:
  forecast.plot()