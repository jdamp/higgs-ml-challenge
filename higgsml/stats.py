import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def confusion_matrix(predictions: pd.Series, labels: pd.Series) -> np.array:
    n_classes = np.unique(labels).sum()
    result = np.zeros((n_classes, n_classes))
    for pred, label in zip(predictions, labels):
        result[pred, label] += 1
    return result


def ams2(s: int, b: int, b_reg: int = 10) -> float:
    """Calculates the median discovery significance according to
    the formula
    AMS2 = sqrt(2*((s+b)*ln(1+s/b)-s))

    Args:
        s (int): _description_
        b (int): _description_

    Returns:
        float: _description_
    """
    print(s, b)
    return np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s))


def classifier_ams2(
    pipeline: Pipeline, data: pd.DataFrame, weights: pd.Series
) -> float:
    predictions = pipeline.predict(data)
    s = np.dot(predictions == 1, weights)
    b = np.dot(predictions == 0, weights)
    return ams2(s, b)


def ams_score(target, pred, weights):
    """Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )
    where b_r = 10, b = background, s = signal, log is natural logarithm.
    s and b are the unnormalised true positive and false positive rates,
    respectively, weighted by the weights of the dataset.
    """
    # true positive rate, weighted

    s = weights.dot(np.logical_and(pred == 1, target == 1))
    # false positive rate, weighted

    b = weights.dot(np.logical_and(pred == 1, target == 0))
    br = 10.0
    radicand = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    if radicand < 0:
        raise Exception("Radicand is negative.")
    else:
        return np.sqrt(radicand)
