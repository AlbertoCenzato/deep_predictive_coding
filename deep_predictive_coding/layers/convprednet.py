from keras import backend as K
from keras.layers import Recurrent, Conv2D, UpSampling2D, MaxPooling2D, Input
from keras.engine import InputSpec
from keras import activations

from ..keras_utils import legacy_prednet_support
from .prednet import PredNetParams


class ConvPredNetParams(PredNetParams):

    def __init__(self, A_filters, R_filters, conv_filters, A_kernels, Ahat_kernels, R_kernels, conv_kernels, **kwargs):
        super(ConvPredNetParams, self).__init__(A_filters, R_filters, A_kernels, Ahat_kernels, R_kernels, **kwargs)
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels



class ConvPredNet(Recurrent):
    """ This model is an extension of Lotter's PredNet model that processes
       data with a CNN before feeding it to PredNet """

    @staticmethod
    def build_from_params(params):
        return ConvPredNet(params.stack_sizes, params.R_stack_sizes, params.conv_filters, params.A_filt_sizes,
                           params.Ahat_filt_sizes, params.R_filt_sizes, params.conv_kernels, **(params.args))

    @legacy_prednet_support
    def __init__(self, A_filters, R_filters, conv_filters, A_kernels, Ahat_kernels, R_kernels, conv_kernels,
                 pixel_max=1., output_mode='error', extrap_start_time=None, data_format=K.image_data_format(),
                 trainable_autoencoder=False, **kwargs):
        self.A_filters = A_filters
        self.nb_prednet_layers = len(A_filters)
        self.nb_conv_layers = len(conv_filters)
        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_prednet_layers) for layer in
                              ['R', 'E', 'A', 'Ahat']]  # FIXME: add conv-deconv recontruction

        assert len(conv_kernels) == self.nb_conv_layers, 'len(conv_filt_sizes) must equal len(conv_stack_sizes)'
        assert len(R_filters) == self.nb_prednet_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        assert len(conv_filters) == len(conv_kernels), 'len(conv_stack_sizes) must equal len(conv_filt_sizes)'
        assert len(A_kernels) == (self.nb_prednet_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        assert len(Ahat_kernels) == self.nb_prednet_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        assert len(R_kernels) == (self.nb_prednet_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)

        self.R_filters = R_filters
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.A_kernels = A_kernels
        self.Ahat_kernels = Ahat_kernels
        self.R_kernels = R_kernels

        self.pixel_max = pixel_max

        self.error_activation = activations.get('relu')
        self.A_activation = activations.get('relu')
        self.LSTM_activation = activations.get('tanh')
        self.LSTM_inner_activation = activations.get('hard_sigmoid')

        self.trainable_autoencoder = trainable_autoencoder

        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None
        self.extrap_start_time = extrap_start_time

        self.data_format = K.image_data_format()
        self.channel_axis = -3 if data_format == 'channels_first' else -1
        self.row_axis = -2 if data_format == 'channels_first' else -3
        self.column_axis = -1 if data_format == 'channels_first' else -2
        super(ConvPredNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5)]

    def get_initial_state(self, x):
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]

        base_initial_state = K.zeros_like(x)  # (samples, timesteps) + image_shape
        non_channel_axis = -1 if self.data_format == 'channels_first' else -2
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = K.sum(base_initial_state, axis=1)  # (samples, nb_channels)

        initial_states = []
        states_to_pass = ['r', 'c', 'e']
        nlayers_to_pass = {u: self.nb_prednet_layers for u in states_to_pass}
        if self.extrap_start_time is not None:
            states_to_pass.append('ahat')  # pass prediction in states so can use as actual for t+1 when extrapolating
            nlayers_to_pass['ahat'] = 1
        for u in states_to_pass:
            for l in range(nlayers_to_pass[u]):
                ds_factor = 2 ** (l + self.nb_conv_layers)  # downsampling factor due to maxpooling
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = self.R_filters[l]
                elif u == 'e':
                    stack_size = 2 * self.A_filters[l]
                elif u == 'ahat':
                    stack_size = self.A_filters[l]
                output_size = stack_size * nb_row * nb_col  # flattened size

                reducer = K.zeros((input_shape[self.channel_axis], output_size))  # (nb_channels, output_size)
                initial_state = K.dot(base_initial_state, reducer)  # (samples, output_size)
                if self.data_format == 'channels_first':
                    output_shp = (-1, stack_size, nb_row, nb_col)
                else:
                    output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = K.reshape(initial_state, output_shp)
                initial_states += [initial_state]

        if K._BACKEND == 'theano':
            from theano import tensor as T
            # There is a known issue in the Theano scan op when dealing with inputs whose shape is 1 along a dimension.
            # In our case, this is a problem when training on grayscale images, and the below line fixes it.
            initial_states = [T.unbroadcast(init_state, 0, 1) for init_state in initial_states]

        if self.extrap_start_time is not None:
            initial_states += [K.variable(0,
                                          int if K.backend() != 'tensorflow' else 'int32')]  # the last state will correspond to the current timestep
        return initial_states

    def compute_output_shape(self, input_shape):
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (
            self.nb_prednet_layers + 1,)  # 1 additional error unit: convolutional-deconvolutional recontruction error
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_prednet_layers + 1,)  # see comment above
        else:  # FIXME: add conv-deconv option
            stack_str = 'R_stack_sizes' if self.output_layer_type == 'R' else 'stack_sizes'
            stack_mult = 2 if self.output_layer_type == 'E' else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            out_nb_row = input_shape[self.row_axis] / 2 ** self.output_layer_num
            out_nb_col = input_shape[self.column_axis] / 2 ** self.output_layer_num
            if self.data_format == 'channels_first':
                out_shape = (out_stack_size, out_nb_row, out_nb_col)
            else:
                out_shape = (out_nb_row, out_nb_col, out_stack_size)

        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

        self.upsample = UpSampling2D(data_format=self.data_format)
        self.pool = MaxPooling2D(data_format=self.data_format)

        # --- build layers ---
        input = Input(shape=input_shape[2:])

        # convolutional layers
        self.encoder_layers = []
        for l in range(self.nb_conv_layers):
            self.encoder_layers.append(
                Conv2D(self.conv_filters[l], self.conv_kernels[l], padding='same', data_format=self.data_format,
                       activation='relu', trainable=self.trainable_autoencoder))

        # prednet layers
        self.prednet_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'a', 'ahat']}
        for l in range(self.nb_prednet_layers):
            for c in ['i', 'f', 'c', 'o']:
                act = self.LSTM_activation if c == 'c' else self.LSTM_inner_activation
                self.prednet_layers[c].append(Conv2D(self.R_filters[l], self.R_kernels[l], padding='same',
                                                     activation=act, data_format=self.data_format))
            act = 'relu' if l == 0 else self.A_activation
            self.prednet_layers['ahat'].append(Conv2D(self.A_filters[l], self.Ahat_kernels[l], padding='same',
                                                      activation=act, data_format=self.data_format))
            if l < self.nb_prednet_layers - 1:
                self.prednet_layers['a'].append(Conv2D(self.A_filters[l + 1], self.A_kernels[l], padding='same',
                                                       activation=self.A_activation, data_format=self.data_format))
        # deconvolutional layers
        self.decoder_layers = []
        self.decoder_layers.append(Conv2D(1, self.conv_kernels[0], padding='same', data_format=self.data_format,
                                          activation='sigmoid', trainable=self.trainable_autoencoder))
        for l in range(self.nb_conv_layers - 1):
            self.decoder_layers.append(Conv2D(self.conv_filters[l], self.conv_kernels[l + 1], padding='same',
                                              data_format=self.data_format, activation='relu',
                                              trainable=self.trainable_autoencoder))

        # --- compute input shapes ---

        self.trainable_weights = []
        if self.data_format == 'channels_first':
            nb_row, nb_col = (input_shape[-2], input_shape[-1])
            input_channels = input_shape[-3]
        else:
            nb_row, nb_col = (input_shape[-3], input_shape[-2])
            input_channels = input_shape[-1]

        # convolutional
        for l in range(self.nb_conv_layers):
            ds_factor = 2 ** l
            nb_channels = self.conv_filters[l - 1] if l > 0 else input_channels
            in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
            if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
            with K.name_scope('layer_conv_' + str(l)):
                self.encoder_layers[l].build(in_shape)
            self.trainable_weights += self.encoder_layers[l].trainable_weights

        # prednet
        for c in sorted(self.prednet_layers.keys()):
            for l in range(len(self.prednet_layers[c])):
                ds_factor = 2 ** (l + self.nb_conv_layers)
                if c == 'ahat':
                    nb_channels = self.R_filters[l]
                elif c == 'a':
                    nb_channels = 2 * self.R_filters[l]
                else:
                    nb_channels = self.A_filters[l] * 2 + self.R_filters[l]
                    if l < self.nb_prednet_layers - 1:
                        nb_channels += self.R_filters[l + 1]
                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
                if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
                with K.name_scope('layer_' + c + '_' + str(l)):
                    self.prednet_layers[c][l].build(in_shape)
                self.trainable_weights += self.prednet_layers[c][l].trainable_weights

        # deconvolutional
        for l in range(self.nb_conv_layers):
            ds_factor = 2 ** l
            nb_channels = self.conv_filters[l] if l < (self.nb_conv_layers - 1) else self.R_filters[0]
            in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
            if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
            with K.name_scope('layer_deconv_' + str(l)):
                self.decoder_layers[l].build(in_shape)
            self.trainable_weights += self.decoder_layers[l].trainable_weights

        self.states = [None] * self.nb_prednet_layers * 3

        if self.extrap_start_time is not None:
            self.t_extrap = K.variable(self.extrap_start_time, int if K.backend() != 'tensorflow' else 'int32')
            self.states += [None] * 2  # [previous frame prediction, timestep]


    def step(self, a, states):
        r_tm1 = states[:self.nb_prednet_layers]
        c_tm1 = states[self.nb_prednet_layers:2 * self.nb_prednet_layers]
        e_tm1 = states[2 * self.nb_prednet_layers:3 * self.nb_prednet_layers]

        if self.extrap_start_time is not None:
            t = states[-1]
            a = K.switch(t >= self.t_extrap, states[-2],
                         a)  # if past self.extrap_start_time, the previous prediction will be treated as the actual

        c = []
        r = []
        e = []

        # Update prednet R units starting from the top
        for l in reversed(range(self.nb_prednet_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_prednet_layers - 1:
                inputs.append(r_up)

            inputs = K.concatenate(inputs, axis=self.channel_axis)
            i = self.prednet_layers['i'][l].call(inputs)
            f = self.prednet_layers['f'][l].call(inputs)
            o = self.prednet_layers['o'][l].call(inputs)
            _c = f * c_tm1[l] + i * self.prednet_layers['c'][l].call(inputs)
            _r = o * self.LSTM_activation(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)

        internal_prediction = self.prednet_layers['ahat'][0].call(r[0])

        # Update deconvolutional units starting from the top
        for l in reversed(range(self.nb_conv_layers)):
            if l == self.nb_conv_layers - 1:
                deconv = internal_prediction

            upsampled = self.upsample.call(deconv)
            deconv = self.decoder_layers[l].call(upsampled)
            if l == 0:
                deconv = K.minimum(deconv, self.pixel_max)
                frame_prediction = deconv

        e_reconstr_up = self.error_activation(frame_prediction - a)
        e_reconstr_down = self.error_activation(a - frame_prediction)
        reconstr_error = K.concatenate((e_reconstr_up, e_reconstr_down), axis=self.channel_axis)

        # Update convolutional units starting from the bottom
        for l in range(self.nb_conv_layers):
            a = self.encoder_layers[l].call(a)
            a = self.pool.call(a)

        # Update prednet feedforward path starting from the bottom
        for l in range(self.nb_prednet_layers):
            if l == 0:
                ahat = internal_prediction
            else:
                ahat = self.prednet_layers['ahat'][l].call(r[l])  # prediction
            e_up = self.error_activation(ahat - a)  # errors
            e_down = self.error_activation(a - ahat)

            e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))

            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'R':
                    output = r[l]
                elif self.output_layer_type == 'E':
                    output = e[l]

            if l < self.nb_prednet_layers - 1:
                a = self.prednet_layers['a'][l].call(e[l])
                a = self.pool.call(a)  # target for next layer

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                all_error = K.mean(K.batch_flatten(reconstr_error), axis=-1, keepdims=True)  # reconstr_error
                for l in range(self.nb_prednet_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    all_error = K.concatenate((all_error, layer_error),
                                              axis=-1)  # layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

        states = r + c + e
        if self.extrap_start_time is not None:
            states += [frame_prediction, t + 1]
        return output, states


    def set_autoencoder_weights(self, encoder, decoder):
        assert self.nb_conv_layers == len(encoder), 'self.nb_conv_layers must equal len(encoder)'
        assert self.nb_conv_layers == len(decoder), 'self.nb_conv_layers must equal len(decoder)'
        for l in range(self.nb_conv_layers):
            self.encoder_layers[l].set_weights(encoder[l])
            self.decoder_layers[l].set_weights(decoder[l])

    # def getAutoencoder(self, input_shape):
    #   input = Input(shape=input_shape)
    #   config = self.encoder_layers[0].get_config()
    #   autoencoder_layers = Conv2D.from_config(config)(input)
    #   for l in range(self.nb_conv_layers-1):
    #      config = self.pool.get_config()
    #      autoencoder_layers = MaxPooling2D.from_config(config)(autoencoder_layers)
    #      config = self.encoder_layers[l+1].get_config()
    #      autoencoder_layers = Conv2D.from_config(config)(autoencoder_layers)
    #
    #   config = self.decoder_layers[-1].get_config()
    #   autoencoder_layers = Conv2D.from_config(config)(autoencoder_layers)
    #   for l in reversed(range(self.nb_conv_layers-1)):
    #      config = self.upsample.get_config()
    #      autoencoder_layers = UpSampling2D.from_config(config)(autoencoder_layers)
    #      config = self.decoder_layers[l].get_config()
    #      autoencoder_layers = Conv2D.from_config(config)(autoencoder_layers)
    #
    #   autoencoder = Model(inputs=input, outputs=autoencoder_layers)
    #   autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    #   return autoencoder

    def get_config(self):
        config = {'A_filters': self.A_filters,
                  'R_filters': self.R_filters,
                  'conv_filters': self.conv_filters,
                  'A_kernels': self.A_kernels,
                  'Ahat_kernels': self.Ahat_kernels,
                  'R_kernels': self.R_kernels,
                  'conv_kernels': self.conv_kernels,
                  'pixel_max': self.pixel_max,
                  'data_format': self.data_format,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode}
        # 'error_activation':      self.error_activation.__name__,
        # 'A_activation':          self.A_activation.__name__,
        # 'LSTM_activation':       self.LSTM_activation.__name__,
        # 'LSTM_inner_activation': self.LSTM_inner_activation.__name__,}
        base_config = super(ConvPredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
