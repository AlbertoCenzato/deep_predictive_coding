from keras.layers import Layer, RNN, Conv2D, ConvLSTM2D, MaxPooling2D, UpSampling2D
from keras import activations, backend
from keras.engine import InputSpec


class PredNetCell(Layer):

   def __init__(self, input_shape, input_length, A_kernel_sizes, R_kernel_sizes, A_filters, Ahat_filters, R_filters,
                pixel_max=1., output_mode='error', extrap_start_time=None, **kwargs):
      super(PredNetCell, self).__init__(**kwargs)
      
      self.nb_layers = len(A_kernel_sizes)
      default_output_modes = ['prediction', 'error', 'all']
      layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']] # FIXME: add conv-deconv recontruction
      assert len(R_kernel_sizes) == self.nb_layers,       'len(R_kernel_sizes) must equal len(A_kernel_sizes)'
      assert len(A_filters) == (self.nb_layers - 1), 'len(A_filters) must equal len(R_kernel_sizes) - 1'
      assert len(Ahat_filters) == self.nb_layers,       'len(Ahat_filters) must equal len(R_kernel_sizes)'
      assert len(R_filters) == (self.nb_layers),     'len(R_filters) must equal len(R_kernel_sizes)'
      assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)

      self.A_kernel_sizes = A_kernel_sizes
      self.Ahat_kernel_sizes = self.A_kernel_sizes
      self.R_kernel_sizes = R_kernel_sizes
      self.A_filters = A_filters
      self.Ahat_filters = Ahat_filters
      self.R_filters = R_filters

      self.error_activation = activations.get('relu')
      self.A_activation = activations.get('relu')
      self.LSTM_activation = activations.get('tanh')
      self.LSTM_inner_activation = activations.get('hard_sigmoid')

      self.pixel_max = pixel_max

      self.output_mode = output_mode
      if self.output_mode in layer_output_modes:
         self.output_layer_type = self.output_mode[:-1]
         self.output_layer_num = int(self.output_mode[-1])
      else:
         self.output_layer_type = None
         self.output_layer_num = None
      self.extrap_start_time = extrap_start_time

      self.data_format = backend.image_data_format()
      self.channel_axis = -3 if self.data_format == 'channels_first' else -1
      self.row_axis     = -2 if self.data_format == 'channels_first' else -3
      self.column_axis  = -1 if self.data_format == 'channels_first' else -2

      nb_row, nb_col = (input_shape[self.row_axis], input_shape[self.column_axis])
      input_channels = input_shape[self.channel_axis]

      self.computeStatesSizes(input_shape, nb_row, nb_col)

      self.input_spec = [InputSpec(ndim=5)]
      self.buildInternalLayers((input_length,) + input_shape)
      

   def computeStatesSizes(self, input_shape, nb_row, nb_col):
      self.state_size = []
      self.state_size.append(1)
      for x in input_shape:   # output size
         self.state_size[0] = self.state_size[0] * x

      for l in range(self.nb_layers):
         ds_factor = 2 ** l
         nb_row_l, nb_col_l = nb_row // ds_factor, nb_col // ds_factor
         if l == 0:
            e_size = self.state_size[0] * 2
         else:
            e_size = nb_row_l * nb_col_l * self.A_filters[l - 1] * 2
         if l < self.nb_layers - 1:
            r_size = nb_row_l * nb_col_l * self.R_filters[l + 1]
            self.state_size.append(e_size + r_size)
         else:
            self.state_size.append(e_size)

   def buildInternalLayers(self, input_shape):
      self.input_spec = [InputSpec(shape=input_shape)]
      
      self.upsample = UpSampling2D(data_format=self.data_format)
      self.pool = MaxPooling2D(data_format=self.data_format)

      # --- build layers ---
      self.prednet_layers = {c: [] for c in ['r', 'a', 'ahat']}
      for l in range(self.nb_layers):
         conv_lstm = ConvLSTM2D(self.R_filters[l], self.R_kernel_sizes[l], padding='same', activation=self.LSTM_activation, 
                                recurrent_activation=self.LSTM_inner_activation, data_format=self.data_format)
         self.prednet_layers['r'].append(conv_lstm)

         act = 'relu' if l == 0 else self.A_activation
         self.prednet_layers['ahat'].append(Conv2D(self.Ahat_filters[l], self.A_kernel_sizes[l], padding='same', 
                                                   activation=act, data_format=self.data_format))
         if l < self.nb_layers - 1:
            self.prednet_layers['a'].append(Conv2D(self.A_filters[l], self.A_kernel_sizes[l + 1], padding='same', 
                                                   activation=self.A_activation, data_format=self.data_format))
      
      # --- compute input shapes ---
      self.trainable_weights = []
      nb_row, nb_col = (input_shape[self.row_axis], input_shape[self.column_axis])
      input_channels = input_shape[self.channel_axis]
         
      for c in sorted(self.prednet_layers.keys()):
         for l in range(len(self.prednet_layers[c])):
            ds_factor = 2 ** l
            if c == 'ahat':
               nb_channels = self.R_filters[l]
            elif c == 'a':
               nb_channels = 2 * self.Ahat_filters[l]
            else:
               nb_channels = self.Ahat_filters[l] * 2 #+ self.R_filters[l]
               if l < self.nb_layers - 1:
                  nb_channels += self.R_filters[l + 1]
            in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
            if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
            with backend.name_scope('layer_' + c + '_' + str(l)):
               self.prednet_layers[c][l].build(in_shape)
            self.trainable_weights += self.prednet_layers[c][l].trainable_weights

      self.states = [None] * self.nb_layers * 2

      if self.extrap_start_time is not None:
         self.t_extrap = backend.variable(self.extrap_start_time, int if backend.backend() != 'tensorflow' else 'int32')
         self.states += [None] * 2  # [previous frame prediction, timestep]


   def call(self, inputs, states):
      r_tm1 = states[:self.nb_layers]
      e_tm1 = states[self.nb_layers:2 * self.nb_layers]

      if self.extrap_start_time is not None:
         t = states[-1]
         a = backend.switch(t >= self.t_extrap, states[-2], a)  # if past self.extrap_start_time, the previous prediction will be treated as the actual

      r = []
      e = []

      # Update prednet R units starting from the top
      for l in reversed(range(self.nb_layers)): 
         inputs = [r_tm1[l], e_tm1[l]]
         if l < self.nb_layers - 1:
            inputs.append(r_up)

         inputs = backend.concatenate(inputs, axis=self.channel_axis)
         _r = self.prednet_layers['r'][l].call(inputs)
         r.insert(0, _r)

         if l > 0:
            r_up = self.upsample.call(_r)

      # Update prednet feedforward path starting from the bottom
      for l in range(self.nb_layers):
         ahat = self.prednet_layers['ahat'][l].call(r[l]) # prediction
         e_up = self.error_activation(ahat - a) # errors
         e_down = self.error_activation(a - ahat)

         e.append(backend.concatenate((e_up, e_down), axis=self.channel_axis))

         if self.output_layer_num == l:
            if self.output_layer_type == 'A':
               output = a
            elif self.output_layer_type == 'Ahat':
               output = ahat
            elif self.output_layer_type == 'R':
               output = r[l]
            elif self.output_layer_type == 'E':
               output = e[l]

         if l < self.nb_layers - 1:
            a = self.prednet_layers['a'][l].call(e[l])
            a = self.pool.call(a)  # target for next layer

      if self.output_layer_type is None:
         if self.output_mode == 'prediction':
            output = frame_prediction
         else:
            #all_error = backend.mean(backend.batch_flatten(reconstr_error), axis=-1,keepdims=True)#reconstr_error
            for l in range(self.nb_layers):
               layer_error = backend.mean(backend.batch_flatten(e[l]), axis=-1, keepdims=True)
               all_error = layer_error if l == 0 else backend.concatenate((all_error, layer_error), axis=-1)
            if self.output_mode == 'error':
               output = all_error
            else:
               output = backend.concatenate((backend.batch_flatten(frame_prediction), all_error), axis=-1)

      states = r + e
      if self.extrap_start_time is not None:
         states += [frame_prediction, t + 1]
      return output, states


class PredNet(RNN):

   def __init__(self, input_shape, input_length, A_kernel_sizes, R_kernel_sizes, A_filters, Ahat_filters, R_filters,
                pixel_max=1., output_mode='error', extrap_start_time=None, **kwargs):
      cell = PredNetCell(input_shape, input_length, A_kernel_sizes, R_kernel_sizes, A_filters, Ahat_filters, R_filters,
                pixel_max, output_mode, extrap_start_time)
      super(PredNet, self).__init__(cell, **kwargs)
      self.input_spec = cell.input_spec
