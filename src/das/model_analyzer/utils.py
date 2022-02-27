"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


import json
import pickle
import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


def pickle_loader(f):
    with open(f, 'rb') as f:
        return pickle.load(f)


def pickle_saver(data, f):
    with open(f, 'wb') as f:
        return pickle.dump(data, f)


def json_loader(f):
    with open(f, 'r') as f:
        return json.load(f)


def json_saver(data, f):
    with open(f, 'w') as f:
        return json.dump(data, f)


class DataCacher:
    CACHER_LOADER = {
        'pickle': pickle_loader,
        'json': json_loader
    }

    CACHER_SAVER = {
        'pickle': pickle_saver,
        'json': json_saver
    }

    def __init__(self, cache_dir: Path, cacher_type: str = 'pickle') -> None:
        self.cache_dir = cache_dir
        self.cacher_type = cacher_type
        self.suffix = f'.{self.cacher_type}'

    def fix_datatypes(self, data):
        for k, v in data.items():
            if isinstance(v, list):
                for idx, x in enumerate(v):
                    if isinstance(x, dict):
                        v[idx] = self.fix_datatypes(x)
                    elif isinstance(x, np.ndarray):
                        v[idx] = x.tolist()
                    elif isinstance(v, torch.Tensor):
                        v[idx] = x.tolist()
                    elif isinstance(x, uuid.UUID):
                        v[idx] = str(x)
            if isinstance(v, dict):
                data[k] = self.fix_datatypes(v)
            if isinstance(v, np.integer):
                data[k] = int(v)
            if isinstance(v, np.floating):
                data[k] = float(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                data[k] = v.tolist()
            elif isinstance(v, uuid.UUID):
                data[k] = str(v)
        return data

    def load_data(self, load_dir, add_suffix=False):
        data = None
        if add_suffix:
            load_dir = f'{load_dir}{self.suffix}'
        if load_dir.exists():
            data = self.CACHER_LOADER[self.cacher_type](load_dir)
        return data

    def load_data_from_cache(self, filename):
        data = None
        load_dir = self.cache_dir / f'{filename}{self.suffix}'
        if load_dir.exists():
            logger.debug(f"Loading {load_dir} from cache...")
            data = self.CACHER_LOADER[self.cacher_type](load_dir)
        return data

    def save_data(self, data, save_dir, add_suffix=False):
        # make the parent directory
        if not save_dir.parent.exists():
            save_dir.parent.mkdir(parents=True)

        if add_suffix:
            save_dir = f'{save_dir}{self.suffix}'

        # fix the datatypes
        if self.cacher_type == 'json':
            self.fix_datatypes(data)

        # save to file
        self.CACHER_SAVER[self.cacher_type](data, save_dir)

    def save_data_to_cache(self, data, filename):
        # cache data here as this is time consuming
        save_dir = self.cache_dir / f'{filename}{self.suffix}'
        logger.debug(f"Saving {save_dir} to cache...")

        # make the parent directory
        if not save_dir.parent.exists():
            save_dir.parent.mkdir(parents=True)

        # fix the datatypes
        if self.cacher_type == 'json':
            self.fix_datatypes(data)

        # save to file
        self.CACHER_SAVER[self.cacher_type](data, save_dir)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
