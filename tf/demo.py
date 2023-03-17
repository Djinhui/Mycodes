import tensorflow as tf
import numpy as np
inputs = tf.random.normal([2, 10, 8])
gru = tf.keras.layers.GRU(4)
output = gru(inputs)
print(output.shape)
print(output)

gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
whole_sequence_output, final_state = gru(inputs)
print(whole_sequence_output.shape)
print(final_state.shape)

print(whole_sequence_output[:,-1,:])
print(final_state)

print(np.allclose(final_state.numpy(), whole_sequence_output[:,-1,:].numpy()))
