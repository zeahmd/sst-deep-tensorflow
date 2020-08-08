import tensorflow as tf
from sst.model import buildModel
from sst.dataset import SSTContainer
import numpy as np
from loguru import logger
import os
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop, Adadelta
from tqdm import tqdm
from time import time
from math import ceil
from utils import setup_tensorboard_dirs, save_model_file, root_and_binary_title
from tensorboardwriter import tensorboard_write_metrics, tensorboard_write_weights, tensorboard_write_grads, tensorboard_write_prf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def get_optimizer(optim='adam'):
    if optim is 'adam':
        return  Adam()
    elif optim is 'adagrad':
        return Adagrad()
    elif optim is 'sgd':
        return SGD()
    elif optim is 'rmsprop':
        return RMSprop()
    elif optim is 'adadelta':
        return Adadelta()
    else:
        logger.error(f"Invalid optim {optim}")
        os._exit(0)


def train_step(model, loss_func, optimizer, x_train, y_train, train_loss_metric, train_accuracy_metric):
    with tf.GradientTape() as tape:
        preds = model(x_train, training=True)
        loss = loss_func(y_train, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss_metric(loss)
    train_accuracy_metric(y_train, preds)
    return grads

def eval_step(model, loss_func, x_eval, y_eval, eval_loss_metric, eval_accuracy_metric):
    preds = model(x_eval)
    loss = loss_func(y_eval, preds)

    eval_loss_metric(loss)
    eval_accuracy_metric(y_eval, preds)
    return preds


def train_epoch(model, loss_func, optimizer, train_dataset, num_train_batches,
                train_loss_metric, train_accuracy_metric):

    avg_grads = [np.zeros(shape=layer.shape, dtype='float32') for layer in model.trainable_variables]
    num_batch = 0

    for (x_train, y_train) in tqdm(train_dataset, total=num_train_batches, desc='train'):
        grads = train_step(model, loss_func, optimizer, x_train, y_train, train_loss_metric, train_accuracy_metric)
        for i in range(len(avg_grads)):
            avg_grads[i] += np.array(grads[i])

        num_batch += 1

    for i in range(len(avg_grads)):
        avg_grads[i] /= num_batch
        avg_grads[i] = avg_grads[i].tolist()

    return grads

def eval_epoch(model, loss_func, eval_dataset, num_batches, eval_loss_metric, eval_accuracy_metric
               , desc):
    y_true, y_pred = list(), list()
    with tqdm(total=num_batches, desc=desc) as pbar:
        for (x_eval, y_eval) in eval_dataset:
            preds = eval_step(model, loss_func, x_eval, y_eval, eval_loss_metric, eval_accuracy_metric)
            pbar.update(1)
            y_eval = tf.argmax(y_eval, axis=1)
            preds = tf.argmax(preds, axis=1)
            y_true += y_eval.numpy().tolist()
            y_pred += preds.numpy().tolist()

    return precision_score(y_true, y_pred, average='macro'),\
           recall_score(y_true, y_pred, average='macro'),\
           f1_score(y_true, y_pred, average='macro'),\
           confusion_matrix(y_true, y_pred, labels=np.sort(np.unique(np.array(y_true))))

def train(name='lstm', root=False, binary=False, epochs=30, batch_size=32, optim='adam', patience=np.inf,
          tensorboard=False, write_weights=False, write_grads=False, save_model=True):


    dataset_container = SSTContainer(root=root, binary=binary)
    train_X, train_Y = dataset_container.data("train")
    dev_X, dev_Y = dataset_container.data("dev")
    test_X, test_Y = dataset_container.data("test")

    logger.info(f"test size: {len(train_X)}, dev size: {len(dev_X)}, test size: {len(test_X)}")


    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_X, dev_Y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y))

    train_dataset = train_dataset.batch(batch_size=batch_size)
    dev_dataset = dev_dataset.batch(batch_size=batch_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    optimizer = get_optimizer(optim=optim)


    if tensorboard:
        setup_tensorboard_dirs(model_name=name)
        current_time = time()
        train_log_dir = './tensorboard_logs/{}/{}/train'.format(name, current_time)
        dev_log_dir = './tensorboard_logs/{}/{}/dev'.format(name, current_time)
        test_log_dir = './tensorboard_logs/{}/{}/test'.format(name, current_time)

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        if write_grads:
            grad_log_dir = './tensorboard_logs/{}/{}/gradient'.format(name, current_time)
            grad_summary_writer = tf.summary.create_file_writer(grad_log_dir)
        if write_weights:
            weight_log_dir = './tensorboard_logs/{}/{}/weight'.format(name, current_time)
            weight_summary_writer = tf.summary.create_file_writer(weight_log_dir)



    loss_func = tf.keras.losses.CategoricalCrossentropy()

    num_classes = 5
    if binary:
        num_classes = 2

    num_train_batches = ceil(len(train_X) / batch_size)
    num_dev_batches = ceil(len(dev_X) / batch_size)
    num_test_batches = ceil(len(test_X) / batch_size)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy', dtype=tf.float32)

    dev_loss = tf.keras.metrics.Mean('dev_loss', dtype=tf.float32)
    dev_accuracy = tf.keras.metrics.CategoricalAccuracy('dev_accuracy', dtype=tf.float32)

    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy', dtype=tf.float32)


    model = buildModel(name=name,
                       word_index=dataset_container.sst_tokenizer_word_index(),
                       vocab_size=dataset_container.vocab_size(),
                       max_sen_len=dataset_container.max_sen_len(),
                       num_classes=num_classes)

    #used to give title to model file...
    phrase_type, label = root_and_binary_title(root, binary)

    best_loss = np.inf
    stopping_step = 0
    try:
        for epoch in range(epochs):
            grads = train_epoch(model, loss_func, optimizer, train_dataset, num_train_batches, train_loss,
                                train_accuracy)

            if tensorboard:
                if write_weights:
                    tensorboard_write_weights(weight_summary_writer, model, epoch)

                if write_grads:
                    tensorboard_write_grads(grad_summary_writer, model, grads, epoch)

                tensorboard_write_metrics(train_summary_writer, train_loss, epoch)
                tensorboard_write_metrics(train_summary_writer, train_accuracy, epoch)

            _, _, _, _ = eval_epoch(model, loss_func, dev_dataset, num_dev_batches, dev_loss, dev_accuracy, 'dev')
            if tensorboard:
                tensorboard_write_metrics(dev_summary_writer, dev_loss, epoch)
                tensorboard_write_metrics(dev_summary_writer, dev_accuracy, epoch)

            test_precision, test_recall, test_f1_score, cm = eval_epoch(model, loss_func,
                                                                        test_dataset, num_test_batches, test_loss,
                                                                        test_accuracy, 'test')
            if tensorboard:
                tensorboard_write_metrics(test_summary_writer, test_loss, epoch)
                tensorboard_write_metrics(test_summary_writer, test_accuracy, epoch)
                tensorboard_write_prf(test_summary_writer, "precision", test_precision, epoch)
                tensorboard_write_prf(test_summary_writer, "recall", test_recall, epoch)
                tensorboard_write_prf(test_summary_writer, "f1_score", test_f1_score, epoch)

            # Print train, dev, test loss
            # Print train, dev, test accuracy
            logger.info(
                f"epoch={epoch+1}, model={name}, train loss={train_loss.result():.4f}, dev loss={dev_loss.result():.4f}, test loss={test_loss.result():.4f}")
            logger.info(f"epoch={epoch+1}, model={name}, train accuracy={train_accuracy.result()*100:.2f},"
                        f" dev accuracy={dev_accuracy.result()*100:.2f},"
                        f" test accuracy={test_accuracy.result()*100:.2f}")
            logger.info(f"epoch={epoch+1}, model={name}, test precision={test_precision*100:.2f},"
                        f" test recall={test_recall*100:.2f},"
                        f" test f1-score={test_f1_score*100:.2f}")
            logger.info(f"epoch={epoch+1}, model={name}, test confusion matrix= \n" + str(cm))

            # Implement early stopping here
            if test_loss.result() < best_loss:
                best_loss = test_loss.result()
                stopping_step = 0
            else:
                stopping_step += 1

            if stopping_step >= patience:
                logger.info("EarlyStopping!")

                save_model_file(model_name=name,
                                model=model,
                                filename="{}_{}_{}_{}.h5".format(name, phrase_type, label, epoch))
                os._exit(1)

            # Reset all metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            dev_loss.reset_states()
            dev_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

        if save_model:
            save_model_file(model_name=name,
                            model=model,
                            filename="{}_{}_{}_{}.h5".format(name, phrase_type, label, epochs))
    except KeyboardInterrupt:
        choice = input("Do you want to save model?[y/n]")
        if choice is 'y':
            logger.info("Saving model!")
            save_model_file(model_name=name,
                            model=model,
                            filename="{}_{}_{}_{}.h5".format(name, phrase_type, label, "interrupted"))
        os._exit(0)
