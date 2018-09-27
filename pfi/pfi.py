from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, r2_score, mean_absolute_error


class PFI:
    """ Permutation feature importance.
    Example:
        pfi = PFI(model=ml_model, xdata=X, ydata=y)
        pfi.compute_importance_score(ml_type='c')
    """
    def __init__(self, model=None, xdata=None, ydata=None, n_shuffles=5):
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

    # def set_model(self, model):
    #     self.model = model
    #
    # def set_data(self, xdata, ydata):
    #     self.xdata = xdata.copy()
    #     self.ydata = ydata.copy()
    #
    # def set_n_shuffles(self, n_shuffles):
    #     self.n_shuffles = n_shuffles

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
        # df[col_set] = np.random.permutation(df[col_set])
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

    def _get_col_sets_to_use(self, col_sets):
        """ col_cets is a list of lists (each list contains col names). """
        cols_unq_req = set()  # set of unique cols that were requested
        for col_set in col_sets:  # get the unique cols that were passed in col_sets
            for col in col_set:
                cols_unq_req.add(col)
        cols_unq = set(self.xdata.columns.tolist())
        cols_other = cols_unq.difference(cols_unq_req)
        col_sets = sorted(col_sets, key=len, reverse=True)  # sort list based on the length of sublists
        col_sets.extend([[c] for c in cols_other])
        col_sets = col_sets
        
        # Use each col only once
        # cols_unq_req = set()   # set of unique cols that were requested in input arg
        # cols_sets_chosen = []  # selected sets of cols for which fi will be computed
        # for i, col_set in enumerate(col_sets[::-1]):
        #     if len(set(col_set).intersection(cols_unq_req)) == 0:
        #         cols_sets_chosen.append(col_set)
        #         for col in col_set:
        #             cols_unq_req.add(col)
         
        return col_sets

    def compute_spread_pfi(self, col_sets=[], verbose=False):
        """ Compute permutation feature importance by calculating the spread (variance) of
        predictions across the different runs, and then averaging the spreads across samples.
        Args:
            col_sets (list of lists) : each sublist/subset contains group of featues/cols names
        Returns:
            fi (df) : feature importance dataframe with columns 'cols' (column names) and 'imp' (relative importance)
        """        
        # Get a subset of columns
        if len(col_sets) == 0:
            self.col_sets = [[c] for c in self.xdata.columns.tolist()]
        else:
            self.col_sets = self._get_col_sets_to_use(col_sets)

        # Create df to store feature importance
        fi = pd.DataFrame(index=range(len(self.col_sets)), columns=['cols', 'imp'])

        # Iter over cols (col per node)
        for i, col_set in enumerate(self.col_sets):
            pred_df = self._shuf_and_pred(col_set)
            fi.loc[i, 'cols'] = ','.join(col_set)
            fi.loc[i, 'imp'] = pred_df.var(axis=1).mean()

            if verbose:
                if i % 100 == 0:
                    print(f'col {i + 1}/{len(self.col_sets)}')

        if fi['imp'].sum() > 0:
            fi['imp'] = fi['imp'] / fi['imp'].sum()  # normalize importance
        self.fi = fi.sort_values('imp', ascending=False).reset_index(drop=True)
        return self.fi

    def compute_score_pfi(self, ml_type, col_sets=[], verbose=False):
        """ Compute permutation feature importance using the mean decrease in accuracy/score (MDA).
        (a) Compute the prediction score for all samples.
        (b) Execute the computation in part (a) n_shuffles times.
        (c) Average the prediction score across n_shuffles.
        Default prediction scores (metrics):
            (1) classification:  f1_score
            (2) regression:      r2_score
        Args:
            ml_type (str) : string that specifies whether it's a classification ('c') or regression ('r') problem
            col_sets (list of lists) : each sublist/subset contains group of featues/cols names
        Returns:
            fi (df) : feature importance dataframe with columns 'cols' (column names) and 'imp' (relative importance)
        """
        # Get a subset of columns
        if len(col_sets) == 0:
            self.col_sets = [[c] for c in self.xdata.columns.tolist()]
        else:
            self.col_sets = self._get_col_sets_to_use(col_sets)

        # Create df that stores feature importance
        fi = pd.DataFrame(index=range(len(self.col_sets)), columns=['cols', 'imp', 'std'])

        # Compute reference/baseline score
        preds = self.model.predict(self.xdata)
        if preds.ndim > 1 and preds.shape[1] > 1:  # if classification, get the class label
            preds = np.argmax(preds, axis=1)

        if self.ydata.ndim > 1 and self.ydata.shape[1] > 1:  # if classification, get the class label
            ydata = np.argmax(self.ydata, axis=1)
        else:
            ydata = self.ydata

        # Classification or regression
        if ml_type == 'c':
            self.ml_type = 'c'
            ref_score = f1_score(y_true=ydata, y_pred=preds, average='micro')
            # ref_score = f1_score(y_true=ydata, y_pred=preds, average='macro')
        elif ml_type == 'r':
            self.ml_type = 'r'
            ref_score = r2_score(y_true=ydata, y_pred=preds)
        else:
            raise ValueError('ml_type must be either `r` or `c`.')

        # Iter over col sets (col set per node)
        for i, col_set in enumerate(self.col_sets):
            fi.loc[i, 'cols'] = ','.join(col_set)
            pred_df = self._shuf_and_pred(col_set)

            if ml_type == 'c':
                score_vec = [f1_score(y_true=ydata, y_pred=pred_df.iloc[:, j], average='micro') for j in range(pred_df.shape[1])]
                # score_vec = [f1_score(y_true=ydata, y_pred=pred_df.iloc[:, j], average='macro') for j in range(pred_df.shape[1])]
            else:
                score_vec = [r2_score(y_true=ydata, y_pred=pred_df.iloc[:, j]) for j in range(pred_df.shape[1])]

            fi.loc[i, 'imp'] = ref_score - np.array(score_vec).mean()
            fi.loc[i, 'std'] = np.array(score_vec).std()

            if verbose:
                if i % 100 == 0:
                    print(f'col {i + 1}/{len(self.col_sets)}')

        self.fi = fi.sort_values('imp', ascending=False).reset_index(drop=True)
        return self.fi

    def plot_fi(self, figsize=(8, 5), plot_direction='h', max_fea_plot=None, title=None, ylabel=None):
        """ Plot feature importance.
        Args:
            plot_direction (str) : direction of the bars (`v` for vertical, `h` for hrozontal)
            max_fea_plot (int) : number of top most important features to plot
        Returns:
            fig : handle for plt figure
        """
        assert hasattr(self, 'fi'), "'fi' attribute is not defined."
        fontsize = 14

        if max_fea_plot and int(max_fea_plot) <= self.fi.shape[0]:
            fi = self.fi.iloc[:int(max_fea_plot), :]
        else:
            fi = self.fi.copy()

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

