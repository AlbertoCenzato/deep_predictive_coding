import os, json


class TrainingParams(object):

    def __init__(self, nb_epoch=30, batch_size=4, learning_rate=0.0, samples_per_epoch=500):
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.samples_per_epoch = samples_per_epoch


# --------------- SETTINGS ----------------------

TRAINING_PARAMS = TrainingParams()

labPC = "C:\\Users\\micheluzzo"
home = os.path.expanduser("~")
data_dir = "Documents\\alberto_cenzato\\bouncing_balls_renders" if home == labPC else "C:\\Users\\Alberto\\Projects\\Datasets\\bouncing_balls"
models_dir = "Documents\\alberto_cenzato\\models" if home == labPC else "C:\\Users\\Alberto\\Projects\\Keras_models\\deep_predictive_coding"
DATA_DIR = os.path.join(home, data_dir)
MODELS_DIR = os.path.join(home, models_dir)

SYNC_DIR = os.path.join(home, "OneDrive\\SyncTesi")


class ExperimentConfiguration(object):

    def __init__(self, balls=1, dof=2, mean_vel=5000, sequence_len=10, train_pred_start=None, train_pred_end=None,
                 infer_pred_start=None, infer_pred_end=None, sequences=6000, occlusion=False, epochs=30,
                 batch_size=16, model_type='prednet', loss_function='prednet_error_loss'):
        self.balls            = balls
        self.dof              = dof
        self.mean_vel         = mean_vel
        self.sequence_len     = sequence_len
        self.train_pred_start = train_pred_start
        self.train_pred_end   = train_pred_end
        self.infer_pred_start = infer_pred_start
        self.infer_pred_end   = infer_pred_end
        self.sequences        = sequences
        self.occlusion        = occlusion
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.loss_function    = loss_function

        self.model_type = model_type

        self.trParams = TRAINING_PARAMS

        self.__datasets_dir = DATA_DIR
        self.__models_dir   = MODELS_DIR

        self.file_name = ""


    @property
    def model_dir(self):
        return os.path.join(self.__models_dir, str(self))


    @property
    def data_dir(self):
        return os.path.join(self.__datasets_dir, str(self))


    def __str__(self):
        string = "balls" + str(self.balls) + "_dof" + str(self.dof) + "_vel" + str(self.mean_vel) + "_l" + str(
            self.sequence_len) + "_n" + str(self.sequences) + "_epoch" + str(self.epochs)
        if self.occlusion:
            string += "_o"

        return string


def as_experiment_configuration(dictionary):
    config = ExperimentConfiguration()
    for property, value in vars(config).items():
        if property in dictionary:
            setattr(config, property, dictionary[property])

    return config


scheduled_tests_folder = os.path.join(SYNC_DIR, "scheduled_tests")
completed_tests_folder = os.path.join(SYNC_DIR, "completed_tests")
results_folder = os.path.join(SYNC_DIR, "results")
TESTS_CONFIG = []
list = os.listdir(scheduled_tests_folder)
files = [f for f in list if os.path.isfile(os.path.join(scheduled_tests_folder, f))]
for file in files:
    absolute_path = os.path.join(scheduled_tests_folder, file)
    with open(absolute_path) as f:
        config = json.load(f, object_hook=as_experiment_configuration)
        config.file_name = absolute_path
        TESTS_CONFIG.append(config)
