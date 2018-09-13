class ModelType(object):
    PREDNET = 'prednet'
    CONV_PREDNET = 'conv_prednet'
    CONCAT_PREDNET = 'concat_prednet'
    AMPLIF_ERROR = 'amplified_error_prednet'
    SINGLE_PIXEL_ACTIVATION = 'single_pixel_activation'  # input is 1 where the centers of the balls are, 0 everywhere else
    STATE_VECTOR = 'state_vector'


class LossFunctions(object):
    PREDNET_ERROR_LOSS = 'prednet_error_loss'
    PIXEL_LOSS = 'pixel_loss'
    DYNAMIC_LOSS = 'dynamic_loss'
