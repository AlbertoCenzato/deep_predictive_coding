import os
import numpy as np

from deep_predictive_coding.model_type import ModelType


def normalize_data(data, model_type):
    if model_type != ModelType.STATE_VECTOR:
        data = data.astype(np.float32) / 255
    #else:
    #    balls = data.shape[2]//4
    #    #position_indexes = [4*x + y for x in range(balls) for y in range(2)]
    #    velocity_indexes = [4*x + y for x in range(balls) for y in range(2,4)]
    #    #data[:,:,position_indexes] /= np.max(np.abs(data[:,:,position_indexes]))
    #    data[:,:,velocity_indexes] /= np.max(np.abs(data[:,:,velocity_indexes]))
    #    center_x = 64/2
    #    center_y = 48/2
    #    data[:,:,::4 ] = (data[:, :, ::4]  - center_x) / center_x
    #    data[:,:,1::4] = (data[:, :, 1::4] - center_y) / center_y
    return data


# TODO: randomize sample loading!
def load_data(folder_path, num_of_samples=-1, dtype=np.uint8):
    """ Loads .npy data from folderPath. Every file is treated as a training sample.
       The shape of the matrix data is inferred from the shape of the first file,
       so be sure all the files have the same shape. """
    files = os.listdir(folder_path)
    if num_of_samples > len(files):
        return
    if num_of_samples == -1:
        num_of_samples = len(files)

    # detect array shape
    test_file = np.load(os.path.join(folder_path, files[0]))
    data = np.empty((num_of_samples,) + test_file.shape, dtype=dtype)

    # load data
    for i in range(num_of_samples):
        data[i, :] = np.load(os.path.join(folder_path, files[i]))
    return data