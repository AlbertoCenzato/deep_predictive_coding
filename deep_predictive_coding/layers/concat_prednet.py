import numpy as np

from keras.layers import Layer
from keras import backend as K

from .prednet import PredNet, PredNetParams


class ConcatPredNetParams(PredNetParams):

    def __init__(self, nb_layers, stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, **kwargs):
        super(ConcatPredNetParams, self).__init__(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes,
                                                  R_filt_sizes, **kwargs)
        self.nb_layers = nb_layers



class ConcatPredNet(Layer):

    @staticmethod
    def build_from_params(params):
        return ConcatPredNet(params.nb_layers, params.stack_sizes, params.R_stack_sizes, params.A_filt_sizes,
                             params.Ahat_filt_sizes, params.R_filt_sizes, **(params.args))


    def __init__(self, nb_layers, stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, output_mode='all',
                 extrap_start_time=None, extrap_end_time=None, data_format=K.image_data_format(), **kwargs):
        self.prednet = []

        self.nb_layers       = nb_layers
        self.stack_sizes     = stack_sizes
        self.R_stack_sizes   = R_stack_sizes
        self.A_filt_sizes    = A_filt_sizes
        self.Ahat_filt_sizes = Ahat_filt_sizes
        self.R_filt_sizes    = R_filt_sizes
        self.output_mode     = output_mode
        self.data_format     = data_format
        self.__extrap_start_time = extrap_start_time
        self.__extrap_end_time   = extrap_end_time

        self.nb_layers_prednet = len(self.stack_sizes)
        for _ in range(self.nb_layers):
            self.prednet.append(PredNet(self.stack_sizes, self.R_stack_sizes, self.A_filt_sizes, self.Ahat_filt_sizes,
                                        self.R_filt_sizes, output_mode='all', extrap_start_time=self.extrap_start_time,
                                        extrap_end_time=self.extrap_end_time, return_sequences=True))

        super(ConcatPredNet, self).__init__(**kwargs)


    @property
    def extrap_start_time(self):
        return self.__extrap_start_time

    @extrap_start_time.setter
    def extrap_start_time(self, time):
        self.__extrap_start_time = time
        for layer in self.prednet:
            layer.extrap_start_time = time

    @property
    def extrap_end_time(self):
        return self.__extrap_end_time

    @extrap_end_time.setter
    def extrap_end_time(self, time):
        self.__extrap_end_time = time
        for layer in self.prednet:
            layer.extrap_end_time = time


    def build(self, input_shape):
        for prednet_layer in self.prednet:
            prednet_layer.build(input_shape)

        super(ConcatPredNet, self).build(input_shape)


    def call(self, x):
        shape = K.shape(x)
        prediction = x
        for l in range(self.nb_layers):
            prednet_output = self.prednet[l].call(prediction)

            error = prednet_output[:,:,-self.nb_layers_prednet:]
            if l == 0:
                final_error = error
            else:
                final_error = K.concatenate([final_error, error])

            prediction = prednet_output[:,:,:-self.nb_layers_prednet]
            if l < self.nb_layers - 1:
                prediction = K.reshape(prediction, shape)
        if self.output_mode == 'prediction':
            output = K.reshape(prediction, shape)
        elif self.output_mode == 'error':
            output = final_error
        else:
            output = K.concatenate([prediction, final_error])
        return output


    def compute_output_shape(self, input_shape):
        nb_layers_prednet = len(self.stack_sizes)
        error_size = self.nb_layers*nb_layers_prednet
        if self.output_mode == 'all':
            output_shape = input_shape[:2] + (np.prod(input_shape[2:]) + error_size,)
        elif self.output_mode == 'error':
            output_shape = input_shape[:2] + (error_size,)
        else:
            output_shape = input_shape
        return output_shape


    def get_config(self):
        config = {'nb_layers':         self.nb_layers,
                  'stack_sizes':       self.stack_sizes,
                  'R_stack_sizes':     self.R_stack_sizes,
                  'A_filt_sizes':      self.A_filt_sizes,
                  'Ahat_filt_sizes':   self.Ahat_filt_sizes,
                  'R_filt_sizes':      self.R_filt_sizes,
                  'output_mode':       self.output_mode,
                  'extrap_start_time': self.extrap_start_time,
                  'extrap_end_time':   self.extrap_end_time,
                  'data_format':       self.data_format}
        base_config = super(ConcatPredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))