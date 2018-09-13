'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os, math
import numpy as np
import imageio
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import Model, model_from_json
from keras.layers import Input, RNN

from deep_predictive_coding.layers.prednet import PredNet as PredNetVanilla
from deep_predictive_coding.layers.amplified_error_prednet import AmplifiedErrorPredNet
from deep_predictive_coding.layers.convprednet import ConvPredNet
from deep_predictive_coding.layers.state_vector_prednet import StateVectorPredNetCell
from deep_predictive_coding.layers.concat_prednet import ConcatPredNet

from data_management import load_data, normalize_data
from settings import TESTS_CONFIG, results_folder

from deep_predictive_coding.model_type import ModelType, LossFunctions


def compute_and_save_mse(X_test, X_hat, scores_file):
    mse_model = np.mean((X_test[:, 1:] - X_hat [:, 1: ]) ** 2)  # look at all timesteps except the first
    mse_prev  = np.mean((X_test[:, 1:] - X_test[:, :-1]) ** 2)
    with open(scores_file, 'w') as f:
        f.write("Model MSE: %f\n" % mse_model)
        f.write("Previous Frame MSE: %f" % mse_prev)
    return mse_model, mse_prev


def image_from_state_vector(state_vector):
    img_shape = (48, 64, 1)
    samples   = state_vector.shape[0]
    nt        = state_vector.shape[1]
    balls     = state_vector.shape[2] // 4
    x = np.zeros((samples, nt) + img_shape, dtype=np.uint8)
    xx, yy = np.mgrid[:img_shape[0], :img_shape[1]]
    for n in range(samples):
        for t in range(nt):
            figure = None
            for i in range(balls):
                center = state_vector[n,t,4*i:4*i+2]  # *64 # pick the position skipping the velocity
                center = (center[1], center[0])  # swap x and y
                circle = (xx - center[0])**2 + (yy - center[1])**2
                circle = (circle < 10).astype(np.uint8)*255
                figure = circle if i == 0 else figure + circle
            x[n,t,:,:,0] = figure

    return x


def plot_predictions(X_test, X_hat, plots_dir, pred_start):
    nt = X_hat.shape[1]
    n_plot = 5

    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plt.figure(figsize=(nt, 2 * aspect_ratio))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]

    if len(X_test.shape) == 5:
        rows = X_test.shape[-3]
        cols = X_test.shape[-2]
    else:
        rows = X_test.shape[-2]
        cols = X_test.shape[-1]

    mean_mse_per_frame_index      = np.zeros(nt)
    mean_mse_prev_per_frame_index = np.zeros(nt)
    for i in plot_idx:
        overlay_images = []
        side_by_side_images = []
        for t in range(nt):
            predicted_frame = X_hat[i, t, :, :, 0]
            actual_frame = X_test[i, t, :, :, 0]

            # compute mean squared error
            mse = np.mean((actual_frame - predicted_frame) ** 2)
            mean_mse_per_frame_index[t] += mse

            # plot ground-truth
            plt.subplot(gs[t])
            plt.imshow(actual_frame, interpolation='none')
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labelleft=False)
            if t == 0:
                plt.ylabel('Actual', fontsize=10)

            # plot prediction
            plt.subplot(gs[t + nt])
            plt.imshow(predicted_frame, interpolation='none')
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labelleft=False)
            if t == 0: plt.ylabel('Predicted', fontsize=10)
            if t > 0:
                plt.xlabel(str("%.4f" % mse), fontsize=10)
                mean_mse_prev_per_frame_index[t] += np.mean((X_test[i, t - 1, :] - X_test[i, t, :]) ** 2)

            # save gif
            predicted_frame_img = (255 * predicted_frame).astype(np.uint8)
            actual_frame_img    = (255 * actual_frame   ).astype(np.uint8)
            overlay_frame = np.zeros((rows, cols, 3), dtype=np.uint8)
            side_by_side_frame = np.zeros((rows, 2 * cols), dtype=np.uint8)
            overlay_frame[:, :, 0] = predicted_frame_img
            overlay_frame[:, :, 1] = actual_frame_img
            side_by_side_frame[:, :cols] = predicted_frame_img
            side_by_side_frame[:, cols:] = actual_frame_img
            if t != 0:
                overlay_images.append(overlay_frame)  # remove first frameto avoid an annoing change in color
            side_by_side_images.append(side_by_side_frame)

        plt.savefig(plots_dir + 'plot_' + str(i) + '.png')
        plt.clf()
        imageio.mimsave(plots_dir + "prediction_overlay" + str(i) + ".gif", overlay_images, fps=8)
        imageio.mimsave(plots_dir + "prediction_side_by_side" + str(i) + ".gif", side_by_side_images, fps=8)

    mean_mse_per_frame_index = mean_mse_per_frame_index / len(plot_idx)
    mean_mse_prev_per_frame_index = mean_mse_prev_per_frame_index / len(plot_idx)

    plt.figure()
    plt.plot(mean_mse_per_frame_index, 'b-', label='mse_predicted')
    plt.plot(mean_mse_prev_per_frame_index, 'r-', label='mse_prev')
    plt.legend()
    plt.savefig(plots_dir + 'mean_mse_' + str(pred_start) + '.png')


def evaluate(config):
    if config.model_type == ModelType.PREDNET or config.model_type == ModelType.SINGLE_PIXEL_ACTIVATION:
        PredNet = PredNetVanilla
    elif config.model_type == ModelType.CONV_PREDNET:
        PredNet = ConvPredNet
    elif config.model_type == ModelType.AMPLIF_ERROR:
        PredNet = AmplifiedErrorPredNet
    elif config.model_type == ModelType.STATE_VECTOR:
        PredNet = RNN
    elif config.model_type == ModelType.CONCAT_PREDNET:
        PredNet = ConcatPredNet

    weights_file = os.path.join(results_folder, str(config) + "\\prednet_weights.hdf5")
    json_file = os.path.join(results_folder, str(config) + "\\prednet_model.json")

    batch_size = 8

    # load test data
    test_set_dir = os.path.join(config.data_dir, "test")
    data_type = 'uint8' if config.model_type != ModelType.STATE_VECTOR else 'float32'
    data = load_data(test_set_dir, dtype=data_type)
    X_test = normalize_data(data, config.model_type)
    nt = X_test.shape[1]

    # Load trained model
    with open(json_file, 'r') as f:
        json_string = f.read()

    # add custom layers definitions
    if config.model_type != ModelType.STATE_VECTOR and config.model_type != ModelType.CONCAT_PREDNET:
        custom_objects = {PredNet.__name__: PredNet}
    elif config.model_type == ModelType.STATE_VECTOR:
        custom_objects = {StateVectorPredNetCell.__name__: StateVectorPredNetCell}
    elif config.model_type == ModelType.CONCAT_PREDNET:
        custom_objects = {PredNetVanilla.__name__: PredNetVanilla, ConcatPredNet.__name__: ConcatPredNet}

    model = model_from_json(json_string, custom_objects=custom_objects)
    model.load_weights(weights_file)

    # Create testing model (to output predictions)
    layer_config = model.layers[1].get_config()
    if config.model_type == ModelType.STATE_VECTOR:
        cell_config = layer_config['cell']['config']
        cell_config['output_mode'] = 'prediction'
        layer_config['cell'] = StateVectorPredNetCell(**cell_config)
        test_prednet = PredNet(weights=model.layers[1].get_weights(), **layer_config)
    else:
        layer_config['output_mode'] = 'prediction'
        test_prednet = PredNet(weights=model.layers[1].get_weights(), **layer_config)
        test_prednet.extrap_start_time = config.infer_pred_start
        test_prednet.extrap_end_time = config.infer_pred_end
    input_shape = list(model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)

    # make predictions
    X_hat = test_model.predict(X_test, batch_size)

    if config.model_type != ModelType.STATE_VECTOR:
        data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat  = np.transpose(X_hat, (0, 1, 3, 4, 2))
        X_test = (X_test*255).astype(np.uint8)
        X_hat  = (X_hat *255).astype(np.uint8)
    else:
        X_test, X_hat = image_from_state_vector(X_test), image_from_state_vector(X_hat)

    # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
    scores_file = os.path.join(results_folder, str(config) + '\\prediction_scores.txt')
    mse_model, mse_prev = compute_and_save_mse(X_test, X_hat, scores_file)

    plots_dir = os.path.join(results_folder, str(config) + '\\prediction_plots\\')
    plot_predictions(X_test, X_hat, plots_dir, config)


def plot_nn_filter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    plt.savefig("C:\\Users\\micheluzzo\\Desktop\\plot.png")


def get_activations(sess, kernel_name, bias_name, input_tensor, stimuli):
    kernels = sess.run(kernel_name, feed_dict={input_tensor: stimuli})
    biases = sess.run(bias_name, feed_dict={input_tensor: stimuli})
    sum = kernels + biases
    activations = np.maximum(sum, np.zeros_like(sum))
    plot_nn_filter(activations)


def evaluate_autoencoder(model_dir, test_data_dir):
    weights_file = os.path.join(model_dir, "autoencoder_weights.hdf5")
    json_file = os.path.join(model_dir, "autoencoder_model.json")

    with open(json_file, 'r') as f:
        json_string = f.read()
    autoencoder = model_from_json(json_string)
    autoencoder.load_weights(weights_file)

    originaldata = load_data(test_data_dir)
    data = originaldata.astype(np.float32) / 255
    data = np.reshape(data, (data.shape[0],) + data.shape[2:])

    decoded = (autoencoder.predict(data) * 255).astype(np.uint8)
    # decoded = decoded.reshape(decoded, (decoded.shape[0], 48,64))

    gs = gridspec.GridSpec(1, 2)
    for i in range(20):
        image = decoded[i, :, :, 0]
        plt.subplot(gs[0])
        plt.imshow(image, interpolation='none')
        plt.subplot(gs[1])
        plt.imshow(originaldata[i, 0, :, :, 0])

        plt.savefig(os.path.join(model_dir, 'plot_' + str(i) + '.png'))
        plt.clf()


if __name__ == "__main__":
    # evaluateAutoencoder("C:\\Users\\Alberto\\Projects\\Keras_models\\autoencoder_bb", "C:\\Users\\Alberto\\Projects\\Datasets\\bouncing_balls\\autoencoder_dataset\\test")

    for config in TESTS_CONFIG:
        print("Evaluating dataset ", str(config))
        evaluate(config)
