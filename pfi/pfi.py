from __future__ import print_function, division

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, r2_score, mean_absolute_error

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils


class PFI:
    """ Permutation feature importance.
    Example:
        pfi = PFI(model=ml_model, xdata=X, ydata=y)
        pfi.compute_importance_score(ml_type='c')
    """
    def __init__(self, model=None, xdata=None, ydata=None, n_shuffles=5, outdir='.'):
        """
        Args:
            model : ML model
            xdata : X data (features)
            ydata : Y data (target)
            n_shuffles : number of times to shuffle each feature
        """
        if model is not None:
            assert hasattr(model, 'predict'), "'model' must have a method 'predict'."
            self.model = model

        if xdata is not None:
            self.xdata = xdata.copy()
            self.xdata.columns = [str(c) for c in self.xdata.columns]

        if ydata is not None:
            self.ydata = ydata.copy()
        
        self.n_shuffles = n_shuffles

        # ============  Create logger  ============
        # https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
        # Logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # create a file handler
        fh = logging.FileHandler(filename=os.path.join(outdir, 'logfile.log'))
        fh.setLevel(logging.INFO)

        # create a logging format
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        self.logger = logger
        # =========================================


    def _shuf_cols(self, df, col_set, seed=None):
        """ Shuffle a group of cols only once and return the updated df.
        Args:
            df : input df
            col_set : a list of column names that are shuffled to shuffle together
        Returns:
            df : updated df
        """
        df = df.copy()
        df[col_set] = df[col_set].sample(n=df.shape[0], axis=0,
                                         replace=False, random_state=seed).values
        return df


    def _shuf_and_pred(self, col_set):
        """ Shuffle a group of cols multiple times. For each shuffle compute predictions.
        The predictions will be used compute the mean and std across predictions.
        Args:
            col_set : list of column names to shuffle together
        Returns:
            pred_df : df of predictions of size [self.xdata.shape[0], (self.n_shuffles].
                      Each value in df corresponds to a single sample prediction (row) for a given shuffle (col).
                      Statistics (mean, std) across shuffles are then computed.
        """
        pred_df = pd.DataFrame(index=range(self.xdata.shape[0]), columns=range(self.n_shuffles))

        for s in range(self.n_shuffles):
            xdata_shf = self._shuf_cols(self.xdata.copy(), col_set=col_set, seed=None).values

            preds = self.model.predict(xdata_shf)
            if preds.ndim > 1 and preds.shape[1] > 1:  # if classification, get the class label
                preds = np.argmax(preds, axis=1)

            pred_df.iloc[:, s] = preds

        return pred_df


    def gen_col_sets(self, th=0.7, toplot=False, figsize=None, verbose=True):
        """ Generate subgroups of mutually correlated features.
        This problem is solved using the graph theory approach.
        First, compute correlation matrix and mask it using a threshold `th`
        This "adjacency matrix" is treated as graph. All the possible cliques
        in the graph are computed, represneting the feature subgroups.

        Args:
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
        # Compute corr matrix
        cor = utils.compute_cor_mat(self.xdata, zero_diag=True, decimals=5)

        # Absolute value of corr matrix
        cor = cor.abs()

        # Mask the corr matrix
        cor = cor.applymap(lambda x: 1 if x > th else 0)

        # Remove uncorrelated features
        idx = (cor.sum(axis=0) > 10**-3).values
        # print(f'Total features removed: {(idx==False).sum()}.')
        cor = cor.iloc[idx, idx]

        # Zeroing out the traignle may speedup the computation
        # mask = np.zeros_like(cor)
        # mask[np.triu_indices_from(mask)] = True
        # cor = cor * mask

        # https://stackoverflow.com/questions/40284774/efficient-way-for-finding-all-the-complete-subgraphs-of-a-given-graph-python
        G = nx.from_pandas_adjacency(cor)
        t0 = time.time()
        self.cliques = [s for s in nx.enumerate_all_cliques(G) if len(s) > 1]
        self.logger.info(f'Time to compute cliques:  {(time.time()-t0)/60:.3f} mins')
        self.logger.info(f'Corr matrix after removing features shape:  {cor.shape}')
        col_sets = self.cliques

        # # Compute col sets from cliques (use all possible cliques)
        # cols_unq_req = set()  # set of unique cols that were requested
        # for col_set in col_sets:  # get the unique cols that were passed in col_sets
        #     for col in col_set:
        #         cols_unq_req.add(col)
        # cols_unq = set(self.xdata.columns.tolist())
        # cols_other = cols_unq.difference(cols_unq_req)
        # col_sets = sorted(col_sets, key=len, reverse=True)  # sort list based on the length of subsets
        # col_sets.extend([[c] for c in cols_other])
        # self.col_sets = col_sets

        # Compute col sets from cliques (use each col only once)
        cols_unq_req = set()   # set of unique cols that were requested in input arg
        cols_sets_chosen = []  # selected sets of cols for which fi will be computed
        for i, col_set in enumerate(col_sets[::-1]):
            if len(set(col_set).intersection(cols_unq_req)) == 0:
                cols_sets_chosen.append(col_set)
                for col in col_set:
                    cols_unq_req.add(col)
        cols_unq = set(self.xdata.columns.tolist())
        cols_other = cols_unq.difference(cols_unq_req)
        cols_sets_chosen = sorted(cols_sets_chosen, key=len, reverse=True)  # sort list based on the length of subsets
        cols_sets_chosen.extend([[c] for c in cols_other])
        self.col_sets = cols_sets_chosen

        if verbose:
            print(f'Corr matrix after removing features shape:  {cor.shape}')
            print(f'Time to compute cliques:  {(time.time()-t0)/60:.2f} min')

        if toplot:
            if figsize is None:
                figsize=(14, 10)
            utils.plot_cor_heatmap(cor, figsize=figsize, full=True)


    # def compute_score_pfi(self, ml_type, col_sets=[], verbose=False):
    def compute_pfi(self, ml_type, verbose=False):
        """ Compute permutation feature importance using both:
        1. MDA/MDS: mean decrease in accuracy/score.
        2. VAR: prediction variance
        
        1. MDA/MDS:
        First, compute the prediction score for all samples (this is the reference/baseline score).
        Then, for each col set:
            (a) Shuffle together the values of the col set n_shuffles times. For each shuffle,
            compute the prediction score (generates a vector of size [1, n_shuffles]).
            (b) Compute the decrease in score (DS) for each shuffle made in part (b) ([1, n_shuffles]).
            (c) Average the DS for all n_shuffles times to compute the mean decrease in score (MDS).
        Default prediction scores (metrics):
            (1) classification:  f1_score
            (2) regression:      r2_score

        2. VAR:
        (a) For each col set, shuffle together the values of the col set n_shuffles times. For each shuffle,
            compute the predictions for all the samples (generates a vector of values: [n_samples, 1]).
            As a result, after shuffling and predicting n_shuffles times, we get a predictions table of
            size [n_samples, n_shuffles].
        (b) Compute the spread (variance/std) of predictions for each sample across the n_shuffles.
            This generates a vector of size [n_samples, 1].
        (c) Aggregate the values from part (b) by computing the mean or median.

        Args:
            ml_type (str) : string that specifies whether it's a classification ('c') or regression ('r') problem
            col_sets (list of lists) : each sublist/subset contains group of featues/cols names
        Returns:
            fi (df) : feature importance dataframe with columns 'cols' (column names) and 'imp' (relative importance)
        """
        # Get a subset of columns
        if hasattr(self, 'col_sets'):
            # col_sets = self._get_col_sets_to_use(col_sets=self.cliques)
            col_sets = self.col_sets
        else:
            col_sets = [[c] for c in self.xdata.columns.tolist()]

        # Create df to store feature importance
        fi_score = pd.DataFrame(index=range(len(col_sets)), columns=['cols', 'imp', 'std'])
        fi_var = pd.DataFrame(index=range(len(col_sets)), columns=['cols', 'imp'])

        # ===============================================================
        #     Compute reference/baseline score (required for MDA/MDS)
        # ===============================================================
        # Compute predictions
        preds = self.model.predict(self.xdata)
        if preds.ndim > 1 and preds.shape[1] > 1:  # if classification, get the class label
            preds = np.argmax(preds, axis=1)

        if self.ydata.ndim > 1 and self.ydata.shape[1] > 1:  # if classification, get the class label
            ydata = np.argmax(self.ydata, axis=1)
        else:
            ydata = self.ydata

        # Compute the reference/baseline score (classification or regression)
        if ml_type == 'c':
            self.ml_type = 'c'
            ref_score = f1_score(y_true=ydata, y_pred=preds, average='micro')
            # ref_score = f1_score(y_true=ydata, y_pred=preds, average='macro')
        elif ml_type == 'r':
            self.ml_type = 'r'
            ref_score = r2_score(y_true=ydata, y_pred=preds)
        else:
            raise ValueError('ml_type must be either `r` or `c`.')
        # ===============================================================

        # Iter over col sets (col set per node)
        t0 = time.time()
        pred = np.zeros((self.xdata.shape[0], len(col_sets), self.n_shuffles))
        for ci, col_set in enumerate(col_sets):
            fi_score.loc[ci, 'cols'] = ','.join(col_set)
            fi_var.loc[ci, 'cols'] = ','.join(col_set)
            # pred_df = self._shuf_and_pred(col_set)
            pred[:,ci,:] = self._shuf_and_pred(col_set)

            # MDA/MDS
            if ml_type == 'c':
                # score_vec = [f1_score(y_true=ydata, y_pred=pred_df.iloc[:, j], average='micro') for j in range(pred_df.shape[1])]
                # score_vec = [f1_score(y_true=ydata, y_pred=pred_df.iloc[:, j], average='macro') for j in range(pred_df.shape[1])]
                score_vec = [f1_score(y_true=ydata, y_pred=pred[:, ci, j], average='micro') for j in range(pred.shape[2])]
            else:
                # score_vec = [r2_score(y_true=ydata, y_pred=pred_df.iloc[:, j]) for j in range(pred_df.shape[1])]
                score_vec = [r2_score(y_true=ydata, y_pred=pred[:, ci, j]) for j in range(pred.shape[1])]

            fi_score.loc[ci, 'imp'] = ref_score - np.array(score_vec).mean()
            fi_score.loc[ci, 'std'] = np.array(score_vec).std()

            # VAR
            # fi_var.loc[ci, 'imp'] = pred_df.var(axis=1).mean()
            fi_var.loc[ci, 'imp'] = np.mean(np.std(pred[:, ci, :], axis=1))

            if verbose:
                if ci % 100 == 0:
                    print(f'col {ci + 1}/{len(col_sets)}')

        self.logger.info(f'Time to compute PFI:  {(time.time()-t0)/60:.3f} mins')
        self.pred = pred

        # MDA/MDS
        self.fi_score = fi_score.sort_values('imp', ascending=False).reset_index(drop=True)

        # VAR
        if fi_var['imp'].sum() > 0:
            fi_var['imp'] = fi_var['imp'] / fi_var['imp'].sum()  # normalize importance
        self.fi_var = fi_var.sort_values('imp', ascending=False).reset_index(drop=True)


    def _plot_fi(self, fi, figsize=(8, 5), plot_direction='h', max_cols=None, title=None, ylabel=None):
        """ Plot feature importance.
        Args:
            plot_direction (str) : direction of the bars (`v` for vertical, `h` for hrozontal)
            max_cols (int) : number of top most important features to plot
        Returns:
            fig : handle for plt figure
        """
        # assert hasattr(self, 'fi'), "'fi' attribute is not defined."
        fontsize = 14

        if max_cols and int(max_cols) <= fi.shape[0]:
            fi = fi.iloc[:int(max_cols), :]

        fig, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)

        if plot_direction=='v':
            if 'std' in fi.columns.tolist():
                ax.bar(range(len(fi)), fi['imp'], yerr=fi['std'], color='b', align='center', ecolor='r')
            else:
                ax.bar(range(len(fi)), fi['imp'], color='b', align='center')
            ax.set_xticks(range(len(fi)))
            ax.set_xticklabels(fi['cols'], rotation='vertical', fontsize=fontsize)
            ax.set_xlim([-1, len(fi)])

            ax.set_xlabel('Feature', fontsize=fontsize)
            ax.set_ylabel('Importance', fontsize=fontsize)
        else:
            if 'std' in fi.columns.tolist():
                ax.barh(range(len(fi)), fi['imp'], yerr=fi['std'], color='b', align='center', ecolor='r')
            else:
                ax.barh(range(len(fi)), fi['imp'], color='b', align='center')
            ax.set_yticks(range(len(fi)))
            ax.set_yticklabels(fi['cols'], rotation='horizontal', fontsize=fontsize)
            ax.set_ylim([-1, len(fi)])

            # ax.invert_yaxis()
            ax.set_ylabel('Feature', fontsize=fontsize)
            ax.set_xlabel('Importance', fontsize=fontsize)

        ax.grid()

        return fig


    def plot_var_fi(self, figsize=(8, 5), plot_direction='h', max_cols=None, title=None, ylabel=None):
        """ Plot fi computed using pfi var. """
        fig = self._plot_fi(fi=self.fi_var, figsize=figsize, plot_direction=plot_direction,
                            max_cols=max_cols, title=title, ylabel=ylabel)
        self.fig_fi_var = fig
        return fig


    def plot_score_fi(self, figsize=(8, 5), plot_direction='h', max_cols=None, title=None, ylabel=None):
        """ Plot fi computed using pfi score. """
        fig = self._plot_fi(fi=self.fi_score, figsize=figsize, plot_direction=plot_direction,
                            max_cols=max_cols, title=title, ylabel=ylabel)
        self.fig_fi_score = fig
        return fig


    def dump(self, path='.', name=None):        
        """ Dump resutls to files.
        https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
        """
        if name:
            var_filename = 'fi_var_' + name + '.csv'
            score_filename = 'fi_score_' + name + '.csv'
            colset_filename = 'colsets_' + name + '.json'
            clique_filename = 'cliques_' + name + '.json'
            pred_filename = 'pred_' + name + '.npy'
        else:
            var_filename = 'fi_var.csv'
            score_filename = 'fi_score.csv'            
            colset_filename = 'colsets.json'
            clique_filename = 'cliques.json'
            pred_filename = 'pred.npy'

        if hasattr(self, 'col_sets'):
            with open(os.path.join(path, colset_filename), 'w') as fh:  
                json.dump(self.col_sets, fh)
            with open(os.path.join(path, clique_filename), 'w') as fh:  
                json.dump(self.cliques, fh)

        if hasattr(self, 'fi_var'):
            self.fi_var.to_csv(os.path.join(path, var_filename), index=False)
        
        if hasattr(self, 'fi_score'):
            self.fi_var.to_csv(os.path.join(path, score_filename), index=False)

        np.save(os.path.join(path, pred_filename), self.pred, allow_pickle=False)
