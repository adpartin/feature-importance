from __future__ import print_function, division

import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx

utils_path = os.path.abspath('utils_py')
sys.path.append(utils_path)
import utils_all as utils


def shuf_cols(df, col_set, seed=None):
    """ Shuffle a group of cols only once and return the updated df. """
    df = df.copy()
    df[col_set] = df[col_set].sample(n=df.shape[0], axis=0,
                                     replace=False, random_state=seed).values
    return df


# def infer_with_col_shuffle_multi(model, xdata, col, n_shuffles):
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


# def infer_with_col_shuffle(model, xdata, col, outbasedir='.'):
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


def get_fea_groups(X, th=0.5, toplot=False, figsize=None):
    """ Generate subgroups of mutually correlated features.
    This problem is solved using the graph theory approach.
    First, compute correlation matrix and mask it using a threshold `th`
    This "adjacency matrix" is treated as graph. All the possible cliques
    in the graph are computed, represneting the feature subgroups.

    Args:
        X : input dataframe
        th : correlation threshold
    Returns:
        cliques (list of lists) : each sublist/subgroup contains group of featues/cols names

    Reference:
    https://stackoverflow.com/questions/40284774/efficient-way-for-finding-all-the-complete-subgraphs-of-a-given-graph-python
    A = np.array([[0, 1, 1, 0, 0],
                  [1, 0, 1, 0, 0],
                  [1, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0]])
    G = nx.from_numpy_matrix(A)
    [s for s in nx.enumerate_all_cliques(G) if len(s) > 1]
    """
    X = X.copy()

    # Compute corr matrix
    cor = utils.compute_cor_mat(X, zero_diag=True, decimals=5)

    # Absolute value of corr matrix
    cor = cor.abs()

    # Mask the corr matrix
    cor = cor.applymap(lambda x: 1 if x > th else 0)
    # cor[cor < th] = 0
    # cor[cor >= th] = 1

    # Remove uncorrelated features
    idx = (cor.sum(axis=0) > 10**-3).values
    # print(f'Total features removed: {(idx==False).sum()}.')
    cor = cor.iloc[idx, idx]
    print(f'cor matrix after removing features shape {cor.shape}')

    # Zeroing out the traignle may speedup the computation
    # mask = np.zeros_like(cor)
    # mask[np.triu_indices_from(mask)] = True
    # cor = cor * mask

    # https://stackoverflow.com/questions/40284774/efficient-way-for-finding-all-the-complete-subgraphs-of-a-given-graph-python
    # https://networkx.github.io/documentation/stable/reference/generated/networkx.convert_matrix.from_numpy_matrix.html
    # https://networkx.github.io/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_adjacency.html
    # G = nx.from_numpy_matrix(cor.values)
    G = nx.from_pandas_adjacency(cor)
    t0 = time.time()
    cliques = [s for s in nx.enumerate_all_cliques(G) if len(s) > 1]
    print(f'Time to compute cliques: {(time.time()-t0)/60:.2f} min')

    if toplot:
        if figsize is None:
            figsize=(14, 10)
        utils.plot_cor_heatmap(cor, figsize=figsize, full=True)

    return cliques



