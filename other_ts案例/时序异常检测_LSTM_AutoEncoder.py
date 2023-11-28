# https://medium.com/@manthapavankumar11/anomaly-detection-in-time-series-data-with-the-help-of-lstm-auto-encoders-5f8affaae7a7

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.models import Sequential, Model

# Define constants
DATA_FILEPATH = 'power_consumption.txt'
TRAIN_SIZE = 800
EXT_LSTM_UNITS = 64
INT_LSTM_UNITS = 32

def load_data(filepath):
    """Load time series data from file."""
    project_data = pd.read_csv(filepath, delimiter=";")
    project_data['timestamp'] = pd.to_datetime(project_data['Date']+' '+project_data['Time'])
    project_data['Global_active_power'] = pd.to_numeric(project_data['Global_active_power'], errors='coerce')
    project_data = project_data[["timestamp", "Global_active_power"]]
    project_data.fillna(value=project_data['Global_active_power'].mean(), inplace=True)
    project_data.isna().sum()
    project_data.info()
    project_data.set_index('timestamp',inplace=True)
    return project_data


def compute_seasonal_decomposition(filepath):
    project_data = pd.read_csv(filepath, delimiter=";")
    project_data['timestamp'] = pd.to_datetime(project_data['Date']+' '+project_data['Time'])
    project_data['Global_active_power'] = pd.to_numeric(project_data['Global_active_power'], errors='coerce')
    project_data = project_data[["timestamp", "Global_active_power"]]
    # Change the default figsize
    rcParams['figure.figsize'] = 12, 8

    # Decompose and plot
    decomposed = seasonal_decompose(project_data, model='additive')
    decomposed.plot()


def split_data(data, train_size):
    train_data = data.iloc[:train_size,:]
    test_data = data.iloc[train_size:,:]
    return train_data, test_data


def normalize_data(train_data, test_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    return mean, std, train_data, test_data


def build_model_v2(input_shape, ext_lstm_unnits, int_lstm_units):
    """Build LSTM Autoencoder model."""
    input_layer = Input(shape=input_shape)
    encoder = LSTM(ext_lstm_unnits, activation='relu', return_sequences=True)(input_layer)
    encoder_1 = LSTM(int_lstm_units, activation='relu')(encoder)
    repeat = RepeatVector(input_shape[0])(encoder_1)
    decoder = LSTM(int_lstm_units, activation='relu', return_sequences=True)(repeat)
    decoder_1 = LSTM(ext_lstm_unnits, activation='relu', return_sequences=True)(decoder)
    output_layer = TimeDistributed(Dense(input_shape[1]))(decoder_1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, train_data):
    history = model.fit(train_data, train_data, epochs=2, batch_size=100, validation_split=0.2)
    return history

def predict(model, test_data):
    """Use LSTM Autoencoder model to predict on test data."""
    predictions = model.predict(test_data)
    return predictions


def calculate_error(test_data, predictions):
    mse = np.mean(np.power(test_data - predictions, 2), axis=1)
    return mse

def detect_anomalies(mse):
    threshold = np.mean(mse) + 3 * np.std(mse)
    anomalies = np.where(mse > np.round(threshold, 2))[0]
    return anomalies


def plot_results(test_data, anomalies):
    """Plot test data with detected anomalies."""
    plt.subplots(figsize=(14, 10))
    plt.plot(test_data)
    plt.plot(anomalies, test_data['Global_active_power'][anomalies], 'ro')
    plt.figure(figsize=(12,4))
    plt.title('Power consumption Anomaly Detection')
    plt.xlabel('Date')
    plt.ylabel('Consumption ')
    plt.grid() 
    plt.show()



#load the data from the file
df = load_data(DATA_FILEPATH)
df.head()

#split the data between train and test set
train_data, test_data = split_data(df, TRAIN_SIZE)

# Normalize data
mean, std, train_data, test_data = normalize_data(train_data, test_data)
print("Mean: ", mean)
print("Standard Deviation: ", std)

# Build model
INPUT_SHAPE = (train_data.shape[1], 1)

# model = build_model(INPUT_SHAPE, LSTM_UNITS)
model = build_model_v2(INPUT_SHAPE, EXT_LSTM_UNITS, INT_LSTM_UNITS)

# Train model
history = train_model(model, train_data)

# Predict on test data
test_predictions = predict(model, test_data)

# Calculate reconstruction error for each time step
mse = calculate_error(test_data, test_predictions.reshape(2074459,1))
print("Mean Square Error = ", mse)

# Detect anomalies in test data
anomalies = detect_anomalies(mse)

# Plot test data with anomalies
plot_results(test_data, anomalies)