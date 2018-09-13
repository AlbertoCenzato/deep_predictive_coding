import numpy as np
import tensorflow as tf
import keras.backend as K

from .prednet import PredNet, PredNetParams


class AmplifiedErrorPredNetParams(PredNetParams):

   def __init__(self, amplification_weight, A_filters, R_filters, A_kernels, Ahat_kernels, R_kernels, **kwargs):
      super(AmplifiedErrorPredNetParams, self).__init__(A_filters, R_filters, A_kernels, Ahat_kernels, R_kernels, **kwargs)
      self.amplification_weight = amplification_weight


class AmplifiedErrorPredNet(PredNet):

   @staticmethod
   def build_from_params(params):
      return AmplifiedErrorPredNet(params.amplification_weight, params.stack_sizes, params.R_stack_sizes, 
                                   params.A_filt_sizes, params.Ahat_filt_sizes, params.R_filt_sizes, **(params.args))

   def __init__(self, amplification_weight, stack_sizes, R_stack_sizes, A_filt_sizes, 
                Ahat_filt_sizes, R_filt_sizes, amplify='first_layer', **kwargs):

      super(AmplifiedErrorPredNet, self).__init__(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, **kwargs)

      self.amplification_weight = amplification_weight
      self.amplify = amplify

      error_amplifications = np.ones((self.nb_layers,))
      if amplify == 'first_layer':
         error_amplifications[0] = amplification_weight
      else:
         error_amplifications = amplification_weight * self.error_amplifications

      self.error_amplifications = [tf.constant(x, dtype=tf.float32) for x in error_amplifications]


   def step(self, a, states):
      r_tm1 = states[:self.nb_layers]
      c_tm1 = states[self.nb_layers:2 * self.nb_layers]
      e_tm1_non_ampl = states[2 * self.nb_layers:3 * self.nb_layers]

      # amplify errors
      e_tm1 = []
      for i in range(self.nb_layers):
         e_tm1.append(self.error_amplifications[i]*e_tm1_non_ampl[i])

      ground_truth = a

      if self.extrap_start_time is not None:
         t = states[-1]
         # The previous prediction will be treated as the actual if t between t_extrap_start and
         # t_extrap_end
         a = K.switch(tf.logical_and(t >= self.t_extrap_start, t < self.t_extrap_end), states[-2], a)  

      c = []
      r = []
      e = []

      # Update R units starting from the top
      for l in reversed(range(self.nb_layers)):
         inputs = [r_tm1[l], e_tm1[l]]
         if l < self.nb_layers - 1:
            inputs.append(r_up)

         inputs = K.concatenate(inputs, axis=self.channel_axis)
         i = self.conv_layers['i'][l].call(inputs)
         f = self.conv_layers['f'][l].call(inputs)
         o = self.conv_layers['o'][l].call(inputs)
         _c = f * c_tm1[l] + i * self.conv_layers['c'][l].call(inputs)
         _r = o * self.LSTM_activation(_c)
         c.insert(0, _c)
         r.insert(0, _r)

         if l > 0:
            r_up = self.upsample.call(_r)

      # Update feedforward path starting from the bottom
      for l in range(self.nb_layers):
         ahat = self.conv_layers['ahat'][l].call(r[l])
         if l == 0:
            ahat = K.minimum(ahat, self.pixel_max)
            frame_prediction = ahat

         # compute errors
         e_up = self.error_activation(ahat - a)
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

         if l < self.nb_layers - 1:
            a = self.conv_layers['a'][l].call(e[l]*self.error_amplifications[l])
            a = self.pool.call(a)  # target for next layer

      if self.output_layer_type is None:
         if self.output_mode == 'prediction':
            output = frame_prediction
         else:
            for l in range(self.nb_layers):
               layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
               all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
            if self.output_mode == 'error':
               output = all_error
            else:
               output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

      states = r + c + e
      if self.extrap_start_time is not None:
          states += [frame_prediction, t + 1]
      return output, states


   def get_config(self):
      config = {'amplification_weight': self.amplification_weight,
                'amplify':              self.amplify}
      base_config = super(AmplifiedErrorPredNet, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))