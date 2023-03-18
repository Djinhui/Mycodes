import tensorflow as tf
import catboost, lightgbm, xgboost
import torch

print(tf.config.list_physical_devices('GPU'))

print(torch.cuda.is_available())
