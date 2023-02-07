import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable
import seaborn as sns
from sklearn.pipeline import Pipeline

from higgsml.types import ArrayType, FrameType
from higgsml.stats import classifier_ams2, amsscore, confusion_matrix


def set_style() -> None:
    """Sets the plotting style for this project"""
    plt.style.use("fivethirtyeight")


def add_axes(plot_func: Callable):
    """Decorator to automatically create a figure and corresponding axis if no argument
    is given for a plotting function with an optional axis argument
    Args:
        plot_func (_type_): _description_
    """

    def plot_wrapper(*args, **kwargs):
        ax = kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()
        return plot_func(*args, **kwargs, ax=ax)

    return plot_wrapper


@add_axes
def plot_corr_heatmap(
    data: FrameType, features: ArrayType, ax: plt.axis = None
) -> None:
    """
    Plots a correlation heatmap for the given features in the dataset
    Args:
        data (pd.DataFrame): The dataset of interest
        features (list): a list of features
        ax (plt.axis, optional): Axis object to plot on. Defaults to None.
    """
    sns.heatmap(data.loc[:, features].corr(), ax=ax, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(features)) + 0.5)
    ax.set_yticks(np.arange(len(features)) + 0.5)

    ax.set_xticklabels(features, rotation=90, fontsize=10)
    ax.set_yticklabels(features, fontsize=10)

    def plot_features(data: pd.DataFrame, features: list) -> None:
        ncols = 2
        nrows = int(np.ceil(len(features) / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(6, nrows * 4))


@add_axes
def plot_bdt_score(
    pipeline: Pipeline, data: FrameType, labels: ArrayType, ax: plt.axes = None
) -> None:
    prop_sig = pipeline.predict_proba(data[labels == "s"])[:, 1]
    prop_bkg = pipeline.predict_proba(data[labels == "b"])[:, 1]

    bins = np.linspace(0, 1, 21)
    ax.hist(
        prop_bkg,
        density=True,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="background",
    )
    ax.hist(
        prop_sig, density=True, bins=bins, histtype="step", linewidth=2, label="signal"
    )
    ax.set(xlabel="BDT output", ylabel="Fraction of events")
    ax.legend()


@add_axes
def plot_confusion_matrix(
    predictions: pd.Series,
) -> None:
    pass


@add_axes
def plot_ams_curve(
    predictions: ArrayType,
    labels: ArrayType,
    weights: ArrayType,
    thresholds: np.array,
    ax=None,
) -> None:
    ams = [amsscore(labels, predictions > thr, weights) for thr in thresholds]
    ax.plot(thresholds, ams)
