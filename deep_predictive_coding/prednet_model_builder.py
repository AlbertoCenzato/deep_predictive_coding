import os
import numpy as np

from keras import backend
from keras.layers import Input, TimeDistributed, Dense, RNN, Reshape, Add, Flatten, Concatenate
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from .model_type import ModelType, LossFunctions
from .layers.utility_layers import AbsDiff, Slice, MultiplicativeInverse, ZerosLike
from .layers.convprednet import ConvPredNet, ConvPredNetParams
from .layers.prednet import PredNet, PredNetParams
from .layers.state_vector_prednet import StateVectorPredNetCell
from .layers.amplified_error_prednet import AmplifiedErrorPredNet, AmplifiedErrorPredNetParams
from .layers.concat_prednet import ConcatPredNet, ConcatPredNetParams


class PredNetModelBuilder(object):
    """ This class is used to simplify the instantiation of complex PredNet models """

    def __init__(self, config):
        # Training parameters
        self.nt = config.sequence_len
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.loss_function = config.loss_function

        self.balls = config.balls

        # Model parameters
        self.n_channels, self.im_height, self.im_width = (1, 48, 64)
        if backend.image_data_format() == 'channels_first':
            self.input_shape = (self.n_channels, self.im_height, self.im_width)
        else:
            self.input_shape = (self.im_height,  self.im_width,  self.n_channels)

        self.__init_params(config)
        self.optimizer = Adam()
        self.loss = 'mean_absolute_error' if config.model_type != ModelType.STATE_VECTOR else 'mean_squared_error'
        self.tb_write_grads = False
        self.tb_write_images = False
        self.trainable_autoencoder = False
        self.autoencoder = None
        self.out_dir = config.model_dir
        self.weights_file = os.path.join(self.out_dir, "prednet_weights.hdf5")
        self.tbLogPath = os.path.join(self.out_dir, "tensorboard_logs")

    @property
    def nt(self):
        return self._nt

    @nt.setter
    def nt(self, value):
        self._nt = value
        self.time_loss_weights = 1. / (self._nt - 1) * np.ones(
            (self._nt, 1))  # equally weight all timesteps except the first
        self.time_loss_weights[0] = 0

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def set_tensorboard_verbosity(self, write_grads, write_images):
        self.tb_write_grads = write_grads
        self.tb_write_images = write_images

    def add_pretrained_autoencoder(self, modelFile, weightsFile):
        with open(modelFile, 'r') as f:
            json_string = f.read()
        self.autoencoder = model_from_json(json_string)
        self.autoencoder.load_weights(weightsFile)


    def build_model(self, model_type):
        """ Builds the net """

        if model_type not in {ModelType.PREDNET, ModelType.CONV_PREDNET, ModelType.AMPLIF_ERROR,
                              ModelType.SINGLE_PIXEL_ACTIVATION, ModelType.STATE_VECTOR, ModelType.CONCAT_PREDNET}:
            raise ValueError("Unknown model type")

        # build prednet layer
        prednet = self.__build_prednet_layer(model_type)

        # build error output layers
        input_shape = (self.nt,) + self.input_shape
        inputs = Input(shape=input_shape)
        prednet_output = prednet(inputs)
        errors_size = 2*self.nb_layers if model_type == ModelType.CONCAT_PREDNET else self.nb_layers
        errors = Slice(2, -errors_size, prednet_output.get_shape()[2].value)(prednet_output)
        predictions = Slice(2, 0, -errors_size, name='prediction_slice_2')(prednet_output)
        final_errors = self.__error_function(inputs, errors, predictions, model_type)

        # add pretrained autoencoder weights
        if model_type == ModelType.CONV_PREDNET and self.autoencoder is not None:
            self.__set_autoencoder_weights(prednet)

        # instantiate callbacks used for training
        callbacks = [TensorBoard(log_dir=self.tbLogPath, histogram_freq=self.epochs / 10,
                                 write_grads=self.tb_write_grads, write_images=self.tb_write_images,
                                 batch_size=self.batch_size),
                     ModelCheckpoint(filepath=self.weights_file, monitor='val_loss', save_best_only=True)]

        # build and compile model
        model = Model(inputs=inputs, outputs=final_errors)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model, callbacks


    def __build_prednet_layer(self, model_type):
        if model_type == ModelType.PREDNET or model_type == ModelType.SINGLE_PIXEL_ACTIVATION:
            prednet = PredNet.build_from_params(self.params)
        elif model_type == ModelType.CONV_PREDNET:
            prednet = ConvPredNet.build_from_params(self.params)
            if self.autoencoder is None and not self.trainable_autoencoder:
                raise ValueError("Pretrained autoencoder not set and autoencoder is not trainable!")
        elif model_type == ModelType.AMPLIF_ERROR:
            prednet = AmplifiedErrorPredNet.build_from_params(self.params)
        elif model_type == ModelType.STATE_VECTOR:
            cell = StateVectorPredNetCell(self.balls * 4, self.nt, self.nb_layers, self.params.extrap_start_time,
                                          self.params.extrap_end_time, self.params.output_mode)
            prednet = RNN(cell, return_sequences=True)
        elif model_type == ModelType.CONCAT_PREDNET:
            prednet = ConcatPredNet.build_from_params(self.params)

        return prednet


    def __set_autoencoder_weights(self, prednet):
        prednet.encoder_layers[0].set_weights(self.autoencoder.layers[1].get_weights())
        prednet.encoder_layers[1].set_weights(self.autoencoder.layers[3].get_weights())
        prednet.decoder_layers[1].set_weights(self.autoencoder.layers[6].get_weights())
        prednet.decoder_layers[0].set_weights(self.autoencoder.layers[8].get_weights())


    def __init_params(self, config):
        A_filters = (1, 10, 30)
        R_filters = A_filters
        conv_filters = (3, 5)

        A_kernels = (3, 3)
        Ahat_kernels = (3, 3, 3)
        R_kernels = (3, 3, 3)
        conv_kernels = (3, 3)

        extrap_start_time = config.train_pred_start
        extrap_end_time = config.train_pred_end

        output_mode = 'all'

        self.nb_layers = len(A_filters) if config.model_type != ModelType.STATE_VECTOR else 2

        if config.model_type == ModelType.AMPLIF_ERROR:
            amplify = 'first_layer'
            amplification_weight = 10

            self.params = AmplifiedErrorPredNetParams(amplification_weight, A_filters, R_filters, A_kernels,
                                                      Ahat_kernels, R_kernels, output_mode=output_mode,
                                                      return_sequences=True, extrap_start_time=extrap_start_time,
                                                      extrap_end_time=extrap_end_time, amplify=amplify)
        elif config.model_type == ModelType.SINGLE_PIXEL_ACTIVATION:
            self.params = ConvPredNetParams(A_filters[:-1], R_filters[:-1], conv_filters[:-1], A_kernels[:-1],
                                            Ahat_kernels[:-1], R_kernels[:-1], conv_kernels[:-1],
                                            output_mode=output_mode, return_sequences=True,
                                            extrap_start_time=extrap_start_time, extrap_end_time=extrap_end_time)
        elif config.model_type == ModelType.CONCAT_PREDNET:
            self.params = ConcatPredNetParams(2, A_filters, R_filters, A_kernels, Ahat_kernels, R_kernels,
                                              output_mode=output_mode, extrap_start_time=extrap_start_time,
                                              extrap_end_time=extrap_end_time)
        else:
            self.params = ConvPredNetParams(A_filters, R_filters, conv_filters, A_kernels, Ahat_kernels, R_kernels,
                                            conv_kernels, output_mode=output_mode, return_sequences=True,
                                            extrap_start_time=extrap_start_time, extrap_end_time=extrap_end_time)


    def __error_function(self, model_input, errors, predictions, model_type):
        weights = self.__compute_loss_weights(model_type)
        with backend.name_scope('error'):
            errors_by_time = TimeDistributed(Dense(1, trainable=False, name='weighted_error_by_layer',
                                                   weights=weights))(errors)  # weighted error by layer
            errors_by_time = Reshape((self.nt,), name='reshape_error_by_time')(errors_by_time)  # it is (batch_size, nt)
            final_errors = Dense(1, weights=[self.time_loss_weights, np.zeros(1)], trainable=False,
                                 name='weighted_error_by_time')(errors_by_time)  # weight errors by time
            if self.loss_function == LossFunctions.PIXEL_LOSS:
                sum_weights = [np.ones((predictions.get_shape()[-1],)), np.zeros(1)]  # [kernel weights, bias weights]
                predictions_activ = TimeDistributed(Dense(1, trainable=False, weights=sum_weights))(predictions)
                predictions_activ = Reshape((self.nt,))(predictions_activ)
                reshaped_input = Reshape((-1, np.prod(self.input_shape)))(model_input)
                input_activ    = TimeDistributed(Dense(1, trainable=False, weights=sum_weights))(reshaped_input)
                input_activ    = Reshape((self.nt,))(input_activ)
                pixel_error    = AbsDiff(name='absolute_value')([input_activ, predictions_activ])
                pixel_error    = Dense(1, trainable=False, weights=[self.time_loss_weights, np.zeros(1)])(pixel_error)
                final_errors   = Add()([final_errors, pixel_error])
            elif self.loss_function == LossFunctions.DYNAMIC_LOSS:
                prev_frame = Flatten(name='flatten_frame_1')(Slice(1, 1, 2, name='slice_frame_1')(predictions))  # start from second frame because first is always all-zero
                static_error = ZerosLike(name='zeros_like')(final_errors)
                frame_motion_weights = [np.ones((predictions.get_shape()[-1], 1)), np.zeros(1)]
                for i in range(2, self.nt):
                    frame = Flatten(name='flatten_frame'+str(i))(Slice(1, i, i + 1, name='slice_frame_'+str(i))(predictions))
                    motion_diff  = AbsDiff(name='frame_diff'+str(i))([frame, prev_frame])
                    frame_motion = Dense(1, trainable=False, weights=frame_motion_weights, name='frame_motion'+str(i))(motion_diff)
                    static_error = Add(name='partial_static_error_sum'+str(i))([static_error, frame_motion])
                    prev_frame = frame
                static_error = MultiplicativeInverse(name='multiplicative_inverse')(static_error)
                final_errors = Add(name='prednet_and_dynamic_error')([final_errors, static_error])

        return final_errors


    def __compute_loss_weights(self, model_type):
        if model_type == ModelType.CONV_PREDNET:
            nb_error_units = self.nb_layers + 1
            layer_loss_weights = np.zeros(nb_error_units)
            used_error_index = 0 if self.trainable_autoencoder else 1
            layer_loss_weights[used_error_index] = 1.
        elif model_type == ModelType.CONCAT_PREDNET:
            nb_error_units = 2*self.nb_layers
            layer_loss_weights = np.zeros(nb_error_units)
            layer_loss_weights[0] = 1.
        else:
            nb_error_units = self.nb_layers
            layer_loss_weights = np.zeros(nb_error_units)
            layer_loss_weights[0] = 1.

        return [np.reshape(layer_loss_weights, (nb_error_units, 1)), np.zeros(1)]
