from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from os.path import join
from os import path
import os
from loguru import logger
import shutil

def setup_tensorboard_dirs(model_name=''):
    if not path.exists(join(os.getcwd(), 'tensorboard_logs/{}'.format(model_name))):
        try:
            os.makedirs(join(os.getcwd(), 'tensorboard_logs/{}'.format(model_name)))
        except FileExistsError:
            pass

    for root, dirs, files in os.walk(join(os.getcwd(), 'tensorboard_logs/{}'.format(model_name))):
        for file in files:
            os.unlink(join(root, file))
        for dir in dirs:
            shutil.rmtree(join(root, dir))


def save_model_file(model_name="", model=None, filename=""):
    if not path.exists(join(os.getcwd(), 'trained/{}'.format(model_name))):
        try:
            os.makedirs(join(os.getcwd(), 'trained/{}'.format(model_name)))
        except FileExistsError:
            pass

    model.save(join(join(os.getcwd(), 'trained/{}'.format(model_name)), filename))
    logger.info("Model has been saved with name: {filename}")

def load_model(model_name=""):
    pass

def root_and_binary_title(root, binary):
    if root:
        phrase_type = 'root'
    else:
        phrase_type = 'all'
    if binary:
        label = 'binary'
    else:
        label = 'fine'
    return phrase_type, label