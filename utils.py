from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from os.path import join
from os import path
import os
from loguru import logger
import shutil
import tensorflow as tf

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

def load_saved_model(filename=""):
    if path.exists(join(os.getcwd(),filename)):
        try:
            return tf.keras.models.load_model(join(os.getcwd(), filename))
        except OSError:
            logger.error("Invalid model file!")
            os._exit(0)
    else:
        logger.error(f"Model for {filename.split('/')[1]} with name {filename.split('/')[2]} dont exist!")
        os._exit(0)


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