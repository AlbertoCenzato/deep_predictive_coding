import tensorflow as tf
import keras


def error_unit(A, A_hat, name):
   diff = tf.subtract(input, Ahat_0)
   return tf.nn.relu(keras.layers.concatenate(diff, tf.negative(diff)), name=name)


def build_graph():
   
   batch_size  = 16
   seq_length  = 20
   im_rows = 48
   im_cols = 64
   im_channels = 1
   image_shape = (im_rows, im_cols, im_channels)
   input_shape = (batch_size, seq_length,) + image_shape

   A_channels    = (im_channels, 5, 10)
   R_channels    = (5,10,30)
   Ahat_channels = A_channels

   kernel_size = 3
   padding = 'same'
   data_format = 'channels_last'
   A_activations = 'relu'

   #initial_error_value = () #?????
   E_0_stored = tf.Variable()
   E_1_stored = tf.Variable()
   E_2_stored = tf.Variable()
   prediction = tf.Variable()

   A_0  = tf.placeholder(tf.float32, shape=input_shape, name='input')
   Ahat_0 = keras.layers.Conv2D(Ahat_channels[0], kernel_size, padding=padding, data_format=data_format, activation=A_activations, name='Ahat_0')

   E_0 = error_unit(input, Ahat_0, 'E_0')
   #tf.Variable(initial_error_value, )
   
   conv = keras.layers.Conv2D(A_channels[1], kernel_size, padding=padding, data_format=data_format, activation=A_activations)(E_0)
   A_1 = keras.layers.MaxPool2D(name='A_1')(conv)
   Ahat_1 = keras.layers.Conv2D(Ahat_channels[1], kernel_size, padding=padding, data_format=data_format, activation=A_activations, name='Ahat_1')

   E_1 = error_unit(A_1, Ahat_1, 'E_1')

   conv = keras.layers.Conv2D(A_channels[2], kernel_size, padding=padding, data_format=data_format, activation=A_activations)(E_1)
   A_2  = keras.layers.MaxPool2D(name='A_2')(conv)
   Ahat_2 = keras.layers.Conv2D(Ahat_channels[2], kernel_size, padding=padding, data_format=data_format, activation=A_activations, name='Ahat_2')

   E_2 = error_unit(A_2, Ahat_2, 'E_2')

   R_2 = keras.layers.ConvLSTM2D(R_channels[2], kernel_size, padding=padding, data_format=data_format, name='R_2')(E_2_stored)

