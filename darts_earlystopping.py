import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import NBEATSModel

# read data
series = AirPassengersDataset().load()

# create training and validation sets:
train, val = series.split_after(pd.Timestamp(year=1957, month=12, day=1))

# normalize the time series
transformer = Scaler()
train = transformer.fit_transform(train)
val = transformer.transform(val)

# any TorchMetric or val_loss can be used as the monitor
torch_metrics = MeanAbsolutePercentageError()

# early stop callback
my_stopper = EarlyStopping(
    monitor="val_MeanAbsolutePercentageError",  # "val_loss",
    patience=5,
    min_delta=0.05,
    mode='min',
)
pl_trainer_kwargs = {"callbacks": [my_stopper]}

# create the model
model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=500,
    torch_metrics=torch_metrics,
    pl_trainer_kwargs=pl_trainer_kwargs)

# use validation set for early stopping
model.fit(
    series=train,
    val_series=val,
)



from pytorch_lightning.callbacks import Callback

class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))


loss_logger = LossLogger()

model = SomeTorchForecastingModel(
    ...,
    nr_epochs_val_period=1,  # perform validation after every epoch
    pl_trainer_kwargs={"callbacks": [loss_logger]}
)

# fit must include validation set for "val_loss"
model.fit(...)