import tensorflow as tf


def tensorboard_write_metrics(summary_writer, metric, epoch):
    with summary_writer.as_default():
        tf.summary.scalar(metric.name, metric.result(), step=epoch)
        summary_writer.flush()


def tensorboard_write_prf(summary_writer, name, value, epoch):
    with summary_writer.as_default():
        tf.summary.scalar(name, value, step=epoch)
        summary_writer.flush()


def tensorboard_write_weights(summary_writer, model, epoch):
    with summary_writer.as_default():
        for weight in model.trainable_variables:
            tf.summary.histogram(weight.name, data=weight.numpy().tolist(), step=epoch)
        summary_writer.flush()


def tensorboard_write_grads(summary_writer, model, grads, epoch):
    with summary_writer.as_default():
        for weight, grad in zip(model.trainable_variables, grads):
            tf.summary.histogram(weight.name, data=grad, step=epoch)
        summary_writer.flush()


def tensorboard_write_cm(summary_writer, img, epoch):
    pass
