import warnings
import numpy as np
from nbeats_keras.model import NBeatsNet as NBeatsKeras
# from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
warnings.filterwarnings(action='ignore', message='Setting attributes')
import matplotlib.pyplot as plt


def main():
    # NBeats： focus on solving the univariate times series point forecasting problem using deep learning
    # At the moment only Keras supports input_dim > 1. In the original paper, input_dim=1
    num_samples, time_steps, input_dim, output_dim = 5000, 10, 1, 1

    for BackendType in [NBeatsKeras]: # [NBeatsKeras, NBeatsPytorch]
        backend = NBeatsKeras(input_dim=1, output_dim=1,backcast_length=time_steps, forecast_length=output_dim,
                              stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
                              nb_blocks_per_stack=2, thetas_dim=(4,4), share_weights_in_stack=True, hidden_layer_units=64)
        
        backend.compile(loss='mse', optimizer='adam')
        
        x = np.random.uniform(size=(num_samples, time_steps, input_dim))
        y = np.mean(x, axis=1, keepdims=True)
        print('x shape:', x.shape, 'y shape:', y.shape) # x shape: (5000, 10, 1) y shape: (5000, 1, 1)

        c = num_samples // 10
        x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
        test_size = len(x_test)

        print('Training...')
        backend.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128)

        # save the model for later
        backend.save('n_beats_model.h5')

        # Predict on the testing set (forecast)
        predictions_forecast = backend.predict(x_test)
        print('predictions_forecast shape',predictions_forecast.shape) # (500, 1, 1)
        np.testing.assert_equal(predictions_forecast.shape, (test_size, backend.forecast_length, output_dim))

        # Predict on the testing set (backcast).
        predictions_backcast = backend.predict(x_test, return_backcast=True)
        np.testing.assert_equal(predictions_backcast.shape, (test_size, backend.backcast_length, output_dim))
        print('predictions_backcast shape',predictions_backcast.shape) # (500, 10, 1)

        # Load the model.
        model_2 = BackendType.load('n_beats_model.h5')

        np.testing.assert_almost_equal(predictions_forecast, model_2.predict(x_test))

        plt.figure(figsize=(12, 7))
        plt.plot(y_test.reshape(-1,)[:30], label='real')
        plt.plot(predictions_forecast.reshape(-1,)[:30], label='predict')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()