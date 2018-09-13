from keras.layers import Lambda
from keras import backend


def Slice(dimension, start, end, name=None):
    """ Python array slicing is supported by Tensorflow but in order to use it
       with Keras it is necessary to wrap the slicing operation in a Keras layer """

    def func(x):
        if dimension == 0:
            return x[start:end]
        elif dimension == 1:
            return x[:, start:end]
        elif dimension == 2:
            return x[:, :, start:end]
        elif dimension == 3:
            return x[:, :, :, start:end]
        elif dimension == 4:
            return x[:, :, :, :, start:end]

    if name is None:
        return Lambda(func)
    return Lambda(func, name=name)


def AbsDiff(name=None):
    def func(tensor_pair):
        diff = tensor_pair[0] - tensor_pair[1]
        return abs(diff)

    if name is None:
        return Lambda(func)
    return Lambda(func, name=name)


def MultiplicativeInverse(name=None):
    def func(tensor):
        # clip to avoid division by 0
        return backend.ones_like(tensor) / backend.clip(tensor, backend.epsilon(), None)

    if name is None:
        return Lambda(func)
    return Lambda(func, name=name)


def ZerosLike(name=None):
    def func(tensor):
        return backend.zeros_like(tensor)

    if name is None:
        return Lambda(func)
    return Lambda(func, name=name)
