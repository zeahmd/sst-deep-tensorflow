from sst.dataset import SSTContainer
from sst.model import buildModel
from training import train
import numpy as np
import os
import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-n", "--name", default="lstm", help="model name")
@click.option("-r", "--root", is_flag=True, help="SST root or all")
@click.option("-b", "--binary", is_flag=True, help="SST binary or fine")
@click.option("-e", "--epochs", default=30, help="no of training iterations/epochs")
@click.option("--batch", default=32, help="batch size")
@click.option("-o", "--optim", default="adam", help="optimizer")
@click.option("-l", "--learningrate", default=1e-3, help="learning rate")
@click.option("-p", "--patience", default=np.inf, help="patience for early stopping")
@click.option("-t", "--tensorboard", is_flag=True, help="enable tensorboard")
@click.option(
    "-w", "--weights", is_flag=True, help="write weights, works if tensorboard enabled"
)
@click.option(
    "-g",
    "--gradients",
    is_flag=True,
    help="write gradients, works if tensorboard enabled",
)
@click.option("-s", "--save", is_flag=True, help="save model")
@click.option("--test", is_flag=True, help="test mode otherwise training by default")
@click.option("-f", "--filename", help="saved keras model file path")
def run(
    name,
    root,
    binary,
    epochs,
    batch,
    optim,
    learningrate,
    patience,
    tensorboard,
    weights,
    gradients,
    save,
    test,
    filename,
):
    """
    SST Details:\n
    -----------\n
    root: only root sentences\n
    all: sentences parsed into phrases\n
    binary: only rows with sentiment negative, positive\n
    fine: negative, partially negative, neutral, partially positive, positive\n

    SST Models: rnn, lstm, gru, bilstm, conv1d

    """
    if not test:
        train(
            name,
            root,
            binary,
            epochs,
            batch,
            optim,
            learningrate,
            patience,
            tensorboard,
            weights,
            gradients,
            save,
        )
    else:
        from testing import test

        test(root, binary, filename)


if __name__ == "__main__":
    run()
