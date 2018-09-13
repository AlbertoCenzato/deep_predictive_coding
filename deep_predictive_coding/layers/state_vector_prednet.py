import tensorflow as tf

import keras.backend as K
from keras import activations
from keras.layers import Layer, Dense


class StateVectorPredNetCell(Layer):

    def __init__(self, input_size, nt, nb_layers=2, extrap_start_time=None, extrap_end_time=None,
                 output_mode='error', **kwargs):
        self.fc_activation = 'relu'

        self.lstm_layer_types = ['i', 'f', 'c', 'o']
        self.layers = {c: [] for c in self.lstm_layer_types + ['a', 'ahat']}

        self.nt                = nt
        self.nb_layers         = nb_layers
        self.output_mode       = output_mode
        self.input_size        = input_size
        self.extrap_start_time = extrap_start_time
        self.extrap_end_time   = extrap_end_time

        if self.extrap_start_time is not None:
            self.t_extrap_start = K.variable(self.extrap_start_time, int if K.backend() != 'tensorflow' else 'int32')
            if self.extrap_end_time is None:
                self.extrap_end_time = nt  # if not specified extrap_end_time is the last frame
            self.t_extrap_end = K.variable(self.extrap_end_time, int if K.backend() != 'tensorflow' else 'int32')

        # compute state sizes
        self.state_size = []
        for s in sorted(['r', 'c', 'e']):
            for l in range(self.nb_layers):
                size = self.input_size * (2**(l+1)) if s == 'e' else self.input_size * (2**l)
                self.state_size.append(size)
        if self.extrap_start_time is not None:
            self.state_size += [self.input_size, 1]

        super(StateVectorPredNetCell, self).__init__(**kwargs)


    def build(self, input_shape):
        if input_shape[1] != self.input_size:
            raise ValueError("input_shape[1] must be equal to input_size declared in the constructor")

        # instantiate layers
        for c in self.layers.keys():
            for l in range(self.nb_layers):
                layer_units = self.input_size * (2**l)
                if c == 'a':
                    if l == 0: continue
                    self.layers[c].append(Dense(units=layer_units, activation=self.fc_activation))  # A
                elif c == 'ahat':
                    if l == 0:
                        self.layers[c].append(Dense(units=layer_units))
                    else:
                        self.layers[c].append(Dense(units=layer_units, activation=self.fc_activation))  # A_hat
                else:
                    act = activations.tanh if c == 'c' else activations.hard_sigmoid
                    self.layers[c].append(Dense(units=layer_units, activation=act))

        # build layers
        self.trainable_weights = []
        for c in sorted(self.layers.keys()):
            for l in range(self.nb_layers):
                if c == 'a':
                    if l >= self.nb_layers-1: continue
                    in_shape = (input_shape[0], self.input_size * (2**(l+1)))
                elif c == 'ahat':
                    in_shape = (input_shape[0], self.input_size * (2**l))
                else:
                    in_size = self.input_size * 3 * (2**l)  # LSTM input coming from error units and LSTM inner recurrent state
                    if l < self.nb_layers - 1:  # if it isn't the last layer add the top-down connection from R_(l+1)
                        in_size += self.input_size * (2**(l+1))
                    in_shape = (input_shape[0], in_size)
                with K.name_scope(c + "_" + str(l)):
                    self.layers[c][l].build(in_shape)
                self.trainable_weights += self.layers[c][l].trainable_weights

        super(StateVectorPredNetCell, self).build(input_shape)


    def call(self, a, states):
        c_tm1 = states[:self.nb_layers]
        e_tm1 = states[  self.nb_layers:2*self.nb_layers]
        r_tm1 = states[2*self.nb_layers:3*self.nb_layers]

        if self.extrap_start_time is not None:
            t = states[-1]
            # The previous prediction will be treated as the actual if t between t_extrap_start and t_extrap_end
            a = K.switch(tf.logical_and(t >= self.t_extrap_start, t < self.t_extrap_end), states[-2], a)

        c = []
        r = []
        e = []

        # Update R units starting from the top
        for l in reversed(range(self.nb_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_layers - 1:
                inputs.append(_r)

            inputs = K.concatenate(inputs)
            i = self.layers['i'][l].call(inputs)
            f = self.layers['f'][l].call(inputs)
            o = self.layers['o'][l].call(inputs)
            _c = f * c_tm1[l] + i * self.layers['c'][l].call(inputs)
            if l == 0:
                _r = o * _c
            else:
                _r = o * activations.tanh(_c)
            c.insert(0, _c)
            r.insert(0, _r)

        # Update feed-forward path starting from the bottom
        for l in range(self.nb_layers):
            ahat = self.layers['ahat'][l].call(r[l])
            if l == 0:
                prediction = ahat

            # compute errors
            e_up   = activations.relu(ahat - a)
            e_down = activations.relu(a - ahat)

            e.append(K.concatenate([e_up, e_down]))

            if l < self.nb_layers - 1:
                a = self.layers['a'][l].call(e[l])

        if self.output_mode == 'prediction':
            output = prediction
        else:
            for l in range(self.nb_layers):
                layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                all_error = layer_error if l == 0 else K.concatenate([all_error, layer_error])
            if self.output_mode == 'error':
                output = all_error
            else:
                output = K.concatenate([prediction, all_error])

        states = c + e + r
        if self.extrap_start_time is not None:
            states += [prediction, t + 1]
        return output, states


    def get_config(self):
        config = {'input_size':        self.input_size,
                  'nt':                self.nt,
                  'nb_layers':         self.nb_layers,
                  'extrap_start_time': self.extrap_start_time,
                  'extrap_end_time':   self.extrap_end_time,
                  'output_mode':       self.output_mode}
        base_config = super(StateVectorPredNetCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
