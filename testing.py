from utils import load_saved_model
from sst.dataset import SSTContainer
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from loguru import logger


def test(root, binary, filename=""):
    model = load_saved_model(filename=filename)

    dataset_container = SSTContainer(root=root, binary=binary)
    test_X, test_Y = dataset_container.data("test")

    pred_Y = model.predict(test_X)
    pred_Y = np.argmax(pred_Y, axis=1)

    accuracy_value = accuracy_score(test_Y, pred_Y)
    precision_value = precision_score(test_Y, pred_Y, average="macro")
    recall_value = recall_score(test_Y, pred_Y, average="macro")
    f1_score_value = f1_score(
        test_Y,
        pred_Y,
        average="macro",
    )
    cm = confusion_matrix(test_Y, pred_Y, labels=np.sort(np.unique(np.array(test_Y))))

    logger.info(
        f"accuracy: {accuracy_value}, precision: {precision_value}, recall: {recall_value}, f1-score: {f1_score_value}"
    )
    logger.info(f"confusion matrix: \n {cm}")
