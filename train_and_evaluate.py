import os, shutil
from keras import backend

import settings
from train_model import train
from evaluate_model import evaluate


for config in settings.TESTS_CONFIG:
    print("Training model ", str(config))
    train(config)
    file = os.path.basename(config.file_name)
    os.rename(config.file_name, os.path.join(settings.completed_tests_folder, file))
    shutil.move(config.model_dir, os.path.join(settings.results_folder, str(config)))
    print("Evaluating model ", str(config))
    evaluate(config)
    backend.clear_session()
