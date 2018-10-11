from __future__ import print_function, division

import os
import sys
import time
import json
import pdb
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, r2_score, mean_absolute_error, brier_score_loss

import sklearn
import keras

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils


class PFI:
    """ Permutation feature importance.
    Example:
        pfi = PFI(model=ml_model, xdata=X, ydata=y)
        pfi.compute_pfi(ml_type='c')
    """
    def __init__(self, model=None, xdata=None, ydata=None, n_shuffles=5, y_enc=None, outdir='.'):
        """
        Args:
            model : ML model
            xdata : X data (features)
            ydata : Y data (target)
            n_shuffles : number of times to shuffle each feature
            y_enc : 
        """
        if model is not None:
            assert hasattr(model, 'predict'), "The 'model' must have the method 'predict'."
            self.model = model

        if xdata is not None:
            self.xdata = xdata.copy()
            self.xdata.columns = [str(c) for c in self.xdata.columns]

        if ydata is not None:
            # If multi-class classification, get only the labels from one-hot
            # self.ydata = self._adjust_target_var(ydata)
            if ydata.ndim > 1 and ydata.shape[1] > 1:
                self.ydata = np.argmax(ydata, axis=1)
            else:
                self.ydata = ydata
            # if isinstance(ydata, pd.DataFrame) is False:
            #     self.ydata = pd.DataFrame(ydata.copy())
            # else:
            #     self.ydata = ydata.copy()
        
        self.n_shuffles = n_shuffles

        if y_enc is not None:
            self.y_enc = y_enc
        

        # ============  Create logger  ============
        # https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
        # Logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # create a file handler
        fh = logging.FileHandler(filename=os.path.join(outdir, 'pfi_logfile.log'))
        fh.setLevel(logging.INFO)

        # create a logging format
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        self.logger = logger
        # =========================================


    def _adjust_target_var(target_var):
        """ Convert multi-dimensional target variable (one-hot encoded or multi-dim array
        or probabilities) into label. """
        # if ref_preds.ndim > 1 and ref_preds.shape[1] > 1:
        #     ref_preds = np.argmax(ref_preds, axis=1)
        if target_var.ndim > 1 and target_var.shape[1] > 1:
            target_var = np.argmax(target_var, axis=1)
        return target_var


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
        pred_df = pd.DataFrame(index=range(self.xdata.shape[0]),
                               columns=range(self.n_shuffles))
        
        for s in range(self.n_shuffles):
            xdata_shf = self._shuf_cols(self.xdata.copy(), col_set=col_set, seed=None).values

            preds = self.model.predict(xdata_shf)
            if preds.ndim > 1 and preds.shape[1] > 1:  # if classification, get the class label
                preds = np.argmax(preds, axis=1)

            pred_df.iloc[:, s] = preds

        return pred_df


    def _shuf_and_pred_proba(self, col_set):
        """
        TODO: NEW!
        Shuffle a group of cols multiple times. For each shuffle compute prediction
        probabilities. Then, extract the probability for the correct/true class.
        Args:
            col_set : list of column names to shuffle together
        Returns:
            pred_df : df of prediction probabilities of size [self.xdata.shape[0], (self.n_shuffles].
                      Each value in df corresponds to a prediction probability for the
                      true label(!) for a given shuffle (col).
        """
        # if the output is not a vector (e.g., `predict` method in keras predicts the
        # output probability for each class), then take the label of the class with
        # the highest probability
        pred_df = pd.DataFrame(index=range(self.xdata.shape[0]),
                               columns=range(self.n_shuffles))
        classes = np.unique(self.ydata)

        for s in range(self.n_shuffles):
            xdata_shf = self._shuf_cols(self.xdata.copy(), col_set=col_set, seed=None).values

            ##preds = self.model.predict_proba(xdata_shf)

            if isinstance(self.model, sklearn.ensemble.RandomForestClassifier):
                preds = self.model.predict_proba(xdata_shf)      
            elif isinstance(self.model, keras.Sequential):
                preds = self.model.predict(xdata_shf)
            else:
                raise ValueError('Model is not supported.')            
 
            for cl in classes:
                #idx = self.ydata.values == cl
                idx = self.ydata == cl
                idx = idx.reshape(-1,)
                pred_df.iloc[idx, s] = preds[idx, cl]

        return pred_df


    def gen_col_sets(self, th=0.9, toplot=False, figsize=None, verbose=True):
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
        self.logger.info(f'Time to compute cliques: {(time.time()-t0)/60:.3f} mins')
        self.logger.info(f'Corr matrix after removing features: ({cor.shape[0]}, {cor.shape[1]})')
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
            print(f'Corr matrix after removing features:  {cor.shape}')
            print(f'Time to compute cliques:  {(time.time()-t0)/60:.2f} min')

        if toplot:
            if figsize is None:
                figsize=(14, 10)
            utils.plot_cor_heatmap(cor, figsize=figsize, full=True)


    # def compute_score_pfi(self, ml_type, col_sets=[], verbose=False):
    def compute_pfi(self, ml_type, verbose=True):
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

        3. FI map:
            TODO: fill in ...

        Args:
            ml_type (str) : string that specifies whether it's a classification ('c') or regression ('r') problem
            col_sets (list of lists) : each sublist/subset contains group of featues/cols names
        Returns:
            fi (df) : feature importance dataframe with columns 'cols' (column names) and 'imp' (relative importance)
        """
        # Use unique cols or combine with col sets
        if hasattr(self, 'col_sets'):
            col_sets = self.col_sets
        else:
            col_sets = [[c] for c in self.xdata.columns.tolist()]

        # Create df to store fi
        fi_score = pd.DataFrame(index=range(len(col_sets)), columns=['cols', 'n', 'imp', 'std'])
        fi_var = pd.DataFrame(index=range(len(col_sets)), columns=['cols', 'n', 'imp'])
        fimap = pd.DataFrame(index=range(self.xdata.shape[0])) #, columns=range(len(col_sets)))
        #fi_score_p = pd.DataFrame(index=range(len(col_sets)),
        #                          columns=['cols', 'n'] + ['imp_c'+str(c) for c in classes])
        

        # =======  Compute reference/baseline score (required for MDA/MDS)  =======
        # If classification, the target values might be onehot encoded.
        # Then, get the true target class labels.
        # if self.ydata.ndim > 1 and self.ydata.shape[1] > 1:
        #     ydata = np.argmax(self.ydata, axis=1)
        #     #ydata = self.ydata.idxmax(axis=1)
        # else:
        #     ydata = self.ydata
        ydata = self.ydata

        # Compute predictions
        ref_preds = self.model.predict(self.xdata)
        # if the output is not a vector (e.g., `predict` method in keras predicts the
        # output probability for each class), then take the label of the class with
        # the highest probability
        # ref_preds = self._adjust_target_var(ref_preds)
        if ref_preds.ndim > 1 and ref_preds.shape[1] > 1:
            ref_preds = np.argmax(ref_preds, axis=1)  # extract class labels
            #ref_preds = ref_preds.idxmax(axis=1)  # extract class labels

        # Compute reference/baseline score (classification or regression)
        if ml_type == 'c':
            self.ml_type = 'c'
            ref_score = f1_score(y_true=ydata, y_pred=ref_preds, average='micro')

            # TODO: NEW! classification proba
            # Sort data by label TODO: this must be done before everything!
            # tmp = pd.concat([self.ydata, self.xdata], axis=1)
            # tmp = tmp.sort_values(by='y', axis=0).reset_index(drop=True)
            # self.ydata = tmp.iloc[:,0].copy()
            # self.xdata = tmp.iloc[:,1:].copy()
            # Compute reference prediction probabilities
            #classes = np.unique(ydata.values)
            classes = np.unique(ydata)
            # ref_preds_p_ = [ref_preds_p[self.ydata.values == cl, cl] for cl in classes]  TODO: single line routine below
            ref_preds_p = self.model.predict_proba(self.xdata)  # [samples, classes]
            ref_preds_p_ = np.zeros((ref_preds_p.shape[0],))    # [samples, ]
            for cl in classes:
                #idx = ydata.values == cl
                idx = ydata == cl
                idx = idx.reshape(-1,)
                ref_preds_p_[idx] = ref_preds_p[idx, cl]

            ref_preds_p_ = ref_preds_p_.reshape(-1, 1)
            # each value in ref_preds_p_ contains the pred probability for the correct label 

        elif ml_type == 'r':
            self.ml_type = 'r'
            ref_score = r2_score(y_true=ydata, y_pred=ref_preds) 

        else:
            raise ValueError('ml_type must be either `r` or `c`.')
        # ===============================================================

        # Iter over col sets (col set per node)
        t0 = time.time()
        pred_arr = np.zeros((self.xdata.shape[0], len(col_sets), self.n_shuffles))
        pred_arr_p = np.zeros((self.xdata.shape[0], len(col_sets), self.n_shuffles))  # TODO: for proba     

        for ci, col_set in enumerate(col_sets):
            col_set_name = ','.join(col_set)
            fi_score.loc[ci, 'cols'] = col_set_name
            fi_var.loc[ci, 'cols'] = col_set_name

            fi_score.loc[ci, 'n'] = len(col_set)
            fi_var.loc[ci, 'n'] = len(col_set)
            
            # pred_df = self._shuf_and_pred(col_set)
            pred_arr[:,ci,:] = self._shuf_and_pred(col_set)

            # MDA/MDS
            if ml_type == 'c':
                # score_vec = [f1_score(y_true=ydata, y_pred=pred_df.iloc[:, j], average='micro') for j in range(pred_df.shape[1])]
                score_vec = [f1_score(y_true=ydata, y_pred=pred_arr[:,ci,j], average='micro') for j in range(pred_arr.shape[2])]

                # TODO: NEW!
                pred_arr_p[:,ci,:] = self._shuf_and_pred_proba(col_set)  # TODO: new for proba

                fimap[col_set_name] = - pred_arr_p[:,ci,:].mean(axis=1, keepdims=True) + ref_preds_p_
            else:
                # score_vec = [r2_score(y_true=ydata, y_pred=pred_df.iloc[:, j]) for j in range(pred_df.shape[1])]
                score_vec = [r2_score(y_true=ydata, y_pred=pred_arr[:,ci,j]) for j in range(pred_arr.shape[2])]

            fi_score.loc[ci, 'imp'] = ref_score - np.array(score_vec).mean()
            fi_score.loc[ci, 'std'] = np.array(score_vec).std()

            # VAR
            # fi_var.loc[ci, 'imp'] = pred_df.var(axis=1).mean()
            fi_var.loc[ci, 'imp'] = np.mean(np.std(pred_arr[:,ci,:], axis=1))

            if verbose:
                if ci % 100 == 0:
                    print(f'col {ci + 1}/{len(col_sets)}')

        self.logger.info(f'Time to compute PFI: {(time.time()-t0)/60:.3f} mins')
        self.pred_arr = np.around(pred_arr, decimals=5)

        # MDA/MDS
        self.fi_score = fi_score.sort_values('imp', ascending=False).reset_index(drop=True)

        # VAR
        if fi_var['imp'].sum() > 0:
            fi_var['imp'] = fi_var['imp'] / fi_var['imp'].sum()  # normalize importance
        self.fi_var = fi_var.sort_values('imp', ascending=False).reset_index(drop=True)

        # TODO: NEW
        # MDA/MDS class proba
        if ml_type == 'c':
            # Store data related to classification proba
            # fimap = - pred_arr_p.mean(axis=2) + ref_preds_p_
            # fimap = pd.DataFrame(fimap, columns=[','.join(c) for c in col_sets])
            fimap = fimap.loc[:, sorted(fimap.columns)]
            self.fimap = fimap
            self.pred_arr_p = np.around(pred_arr_p, decimals=5)  # shuffling predictions (entire 3-d data)
            self.ref_preds_p_ = ref_preds_p_  # reference predictions


    def _plot_fi(self, fi, figsize=(8, 5), plot_direction='h', max_cols_plot=None, title=None):
        """ Plot feature importance.
        Args:
            plot_direction (str) : direction of the bars (`v` for vertical, `h` for hrozontal)
            max_cols_plot (int) : number of top most important features to plot
        Returns:
            fig : handle for plt figure
        """
        # assert hasattr(self, 'fi'), "'fi' attribute is not defined."
        fontsize=14
        alpha=0.7

        if max_cols_plot and int(max_cols_plot) <= fi.shape[0]:
            fi = fi.iloc[:int(max_cols_plot), :]

        fig, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)

        if plot_direction=='v':
            if 'std' in fi.columns.tolist():
                ax.bar(range(len(fi)), fi['imp'], color='b', align='center', alpha=alpha, yerr=fi['std'], ecolor='r')
            else:
                ax.bar(range(len(fi)), fi['imp'], color='b', align='center', alpha=alpha)
            ax.set_xticks(range(len(fi)))
            ax.set_xticklabels(fi['cols'], rotation='vertical', fontsize=fontsize)
            ax.set_xlim([-1, len(fi)])
            ax.set_xlabel('Feature', fontsize=fontsize)
            ax.set_ylabel('Importance', fontsize=fontsize)
        else:
            if 'std' in fi.columns.tolist():
                ax.barh(range(len(fi)), fi['imp'], color='b', align='center', alpha=alpha, yerr=fi['std'], ecolor='r')
            else:
                ax.barh(range(len(fi)), fi['imp'], color='b', align='center', alpha=alpha)
            ax.set_yticks(range(len(fi)))
            ax.set_yticklabels(fi['cols'], rotation='horizontal', fontsize=fontsize)
            ax.set_ylim([-1, len(fi)])
            # ax.invert_yaxis()
            ax.set_ylabel('Feature', fontsize=fontsize)
            ax.set_xlabel('Importance', fontsize=fontsize)

        # ax.grid()
        # plt.tight_layout()

        return fig


    def plot_var_fi(self, figsize=(8, 5), plot_direction='h', max_cols_plot=None, title=None):
        """ Plot fi computed using pfi var. """
        fig = self._plot_fi(fi=self.fi_var, figsize=figsize, plot_direction=plot_direction,
                            max_cols_plot=max_cols_plot, title=title)
        self.fig_fi_var = fig
        return fig


    def plot_score_fi(self, figsize=(8, 5), plot_direction='h', max_cols_plot=None, title=None):
        """ Plot fi computed using pfi score. """
        fig = self._plot_fi(fi=self.fi_score, figsize=figsize, plot_direction=plot_direction,
                            max_cols_plot=max_cols_plot, title=title)
        self.fig_fi_score = fig
        return fig


    def plot_fimap(self, figsize=(8, 5), n_top_cols=10, title=None, drop_correlated=True):
        """ Plot heatmap of fi per class.
        TODO: NEW! Not finished
        """
        if hasattr(self, 'fimap'):
            fimap = self.fimap
        else:
            pass  # TODO:
        
        fontsize=14
        ydata = self.ydata
        y_enc = self.y_enc

        if drop_correlated:
            colsets_to_drop = [c for c in fimap.columns.tolist() if (len(c.split(',')) > 1)]
            fimap = fimap.drop(columns=colsets_to_drop)

        # Add the label code and label name columns
        # TODO: make it more general (list or array)
        y_enc_dict = {y_enc.loc[i, 'code']: y_enc.loc[i, 'label'] for i in range(len(y_enc))}
        fimap.insert(loc=0, column='y', value=ydata)
        fimap.insert(loc=1, column='label', value=[y_enc_dict[c] for c in ydata])

        # Create dict that contains fimap per label
        fimap_label_dict = {c: fimap[ydata==c].reset_index(drop=True) for c in np.unique(fimap['y'])}

        # Extract the n_top_cols most important features per label 
        n_top_cols = 10      # get this number of top most important cols per label
        top_cols_union = []  # contains the union of top most important cols per label
        df_fi_top_cols = pd.DataFrame(index=range(len(fimap_label_dict)),
                                      columns=['y', 'label'] + ['f'+str(n) for n in range(len(fimap_label_dict))])

        for i, (c, mp) in enumerate(fimap_label_dict.items()):
            fi_tmp = mp.iloc[:,2:].median(axis=0).sort_values(ascending=False)
            #fi_tmp = mp.iloc[:,2:].mean(axis=0).sort_values(ascending=False)
            #fi_tmp = mp.iloc[:,2:].sum(axis=0).sort_values(ascending=False)
            top_cols_union.extend(fi_tmp[:n_top_cols].index)
            df_fi_top_cols.loc[i,'y'] = c
            df_fi_top_cols.loc[i,'label'] = y_enc_dict[c]
            df_fi_top_cols.iloc[i,2:] = fi_tmp[:n_top_cols].index

        top_cols_union = set(top_cols_union)

        # Keep the top_cols in the fimap
        fimap_top = fimap[['y', 'label'] + sorted(list(top_cols_union))].copy()        

        # Average importance for each feature and label
        fimap_top_label = fimap_top.groupby(by=['y', 'label']).agg(np.median).reset_index()
        # fimap_top_label = fimap_top.groupby(by=['y', 'type']).agg(np.mean).reset_index()
        # fimap_top_label = fimap_top.groupby(by=['y', 'type']).agg(np.sum).reset_index()

        # Prepare df for plot
        map_plot = fimap_top_label.drop(columns='y')
        map_plot = map_plot.set_index('label')
        map_plot.index.name = ''

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(map_plot, cmap='Blues', linewidths=0.1, linecolor='white', xticklabels=map_plot.columns)
        # ax.set_xticks(range(len(map_plot.columns)))
        # ax.set_xticklabels(map_plot.columns, rotation='vertical', fontsize=fontsize)
        [tick.label.set_fontsize(fontsize) for tick in ax.yaxis.get_major_ticks()]
        [tick.label.set_fontsize(8) for tick in ax.xaxis.get_major_ticks()]
        
        if title:
            ax.set_title(title)

        self.fig_fimap = fig
        return fig        


    def dump(self, path='.', name=None):        
        """ Dump resutls to files.
        https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
        """
        if name:
            var_filename = name + '_fi_var' + '.csv'
            score_filename = name + '_fi_score' + '.csv'
            score_pmap_filename = name + '_fimap' + '.csv'  # TODO: NEW
            colset_filename = name + '_colsets' + '.json'
            clique_filename = name + '_cliques' + '.json'
            pred_filename = name + '_pred'
        else:
            var_filename = 'fi_var.csv'
            score_filename = 'fi_score.csv'
            score_pmap_filename = 'fimap.csv'  # TODO: NEW
            colset_filename = 'colsets.json'
            clique_filename = 'cliques.json'
            pred_filename = 'pred'

        if hasattr(self, 'col_sets'):
            with open(os.path.join(path, colset_filename), 'w') as fh:  
                json.dump(self.col_sets, fh)
            with open(os.path.join(path, clique_filename), 'w') as fh:  
                json.dump(self.cliques, fh)

        if hasattr(self, 'fi_var'):
            self.fi_var.to_csv(os.path.join(path, var_filename), index=False)
        
        if hasattr(self, 'fi_score'):
            self.fi_score.to_csv(os.path.join(path, score_filename), index=False)
        
        # TODO: NEW
        if hasattr(self, 'fimap'):
            self.fimap.to_csv(os.path.join(path, score_pmap_filename), index=False)            

        # np.save(os.path.join(path, pred_filename+'.npy'), self.pred, allow_pickle=False)
        # np.savez_compressed(os.path.join(path, pred_filename+'.npz'), self.pred)
