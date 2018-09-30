from __future__ import print_function, division

import os
import sys
import time
import numpy as np
import pandas as pd

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils


def shuf_cols(df, col_set, seed=None):
    """ Shuffle a group of cols only once and return the updated df. """
    df = df.copy()
    df[col_set] = df[col_set].sample(n=df.shape[0], axis=0,
                                     replace=False, random_state=seed).values
    return df


def shuf_and_pred_multi(model, xdata, col_set, n_shuffles):
    """ Shuffle a group of cols multiple times. For each shuffle compute predictions.
    The predictions will be used compute the mean and std across predictions.
    Args:
        col_set : list of column names to shuffle together    
    Returns:
        pred_df : df of predictions of size [self.xdata.shape[0], (self.n_shuffles].
                  Each value in df corresponds to a single sample prediction (row) for a given shuffle (col).
                  Statistics (mean, std) across shuffles are then computed.
    """
    pred_df = pd.DataFrame(index=range(xdata.shape[0]), columns=range(n_shuffles))

    for s in range(n_shuffles):
        xdata_shf = shuf_cols(xdata.copy(), col_set=col_set, seed=None).values

        preds = model.predict(xdata_shf)
        if preds.ndim > 1 and preds.shape[1] > 1:  # if classification, get the class label
            preds = np.argmax(preds, axis=1)

        pred_df.iloc[:, s] = preds

    return pred_df


def shuf_and_pred(model, xdata, col, outbasedir='.'):
    """ Shuffle the input col only once and make predictions.
    Returns:
        preds : vector of predictions of size [len(self.xdata), 1].
    """
    xdata = xdata.copy()
    xdata_shf = shuf_col(xdata, col=col, seed=None)

    preds = model.predict(xdata_shf)
    if preds.ndim > 1 and preds.shape[1] > 1:  # if classification, get the class
        preds = np.argmax(preds, axis=1)

    # outdir = os.path.join(outbasedir, str(col))
    # make_dir(outdir)
    # pred_df.to_csv(os.path.join(outdir, 'col.' + str(col)), sep='\t', index=False)

    return preds


def aggregate_preds(preddir):
    """
    Compute feature importance.
    Load predictions tables. Each table of size [n_samples, n_shuffles] corresponds to the shuffles performed for that
    specific columns.
    Args:

    Returns:

    """
    pass

