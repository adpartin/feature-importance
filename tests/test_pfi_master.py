"""
This code:
* Imitates pfi computation on a distributed machine.
"""
from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import np_utils

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils

DATAPATH = os.path.join(file_path, 'data', 'data_classification_corr')
MODELDIR = os.path.join(file_path, 'keras_model_classifier')
TEMPDIR = os.path.join(file_path, 'tempdir')
OUTDIR = os.path.join(file_path, 'results_test_pfi_master')
CORR_THRES = 0.74
SEED = 0


def init_params():
    parser = argparse.ArgumentParser(description='Create, train, and store keras model.')
    parser.add_argument('-d', '--datapath', dest='datapath',
                        default=DATAPATH,
                        help='full path to the data')
    parser.add_argument('-m', '--modeldir', dest='modeldir',
                        default=MODELDIR,
                        help='dir of the trained ML model')
    parser.add_argument('-t', '--tempdir', dest='tempdir',
                        default=TEMPDIR,
                        help='dir to store the (temp) predictions for the current col shuffle')
    parser.add_argument('-ns', '--n_shuffles', dest='n_shuffles',
                        default=20, type=int,
                        help='number of shuffles')
    return parser.parse_args()


def run(args):
    print(args)
    datapath = args.datapath
    modeldir = args.modeldir
    tempdir = args.tempdir
    n_shuffles = args.n_shuffles
    corr_th = CORR_THRES

    # Create necessary dirs
    utils.make_dir(TEMPDIR)  # os.makedirs(TEMPDIR, exist_ok=True)  # python 3
    utils.make_dir(OUTDIR)   # os.makedirs(OUTDIR, exist_ok=True)   # python 3
    

    # =======  Load dataset  =======
    data = pd.read_csv(datapath, sep='\t')
    xdata = data.iloc[:, 1:].copy()
    ydata = data.iloc[:, 0].copy()
    features = xdata.columns

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])
    print('np.unique(ydata)', np.unique(ydata))

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    # Split data
    if 'classification' in datapath.split(os.sep)[-1]:
        print('classification')
        xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, stratify=ydata)
    elif 'regression' in datapath.split(os.sep)[-1]:
        print('regression')
        xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED)


    # =======  Load trained keras model  =======
    # (no need to compile the model for inference)
    model_name = 'keras_model.json'
    weights_name = 'keras_weights.h5'

    model_path = os.path.join(modeldir, model_name)
    weights_path = os.path.join(modeldir, weights_name)

    print(f'\nLoading model from ... {model_path}')
    with open(model_path, 'r') as json_file:
        model = json_file.read()
    keras_model = model_from_json(model)
    keras_model.name = 'trained_model'

    # Load weights into new model
    print('\nLoading model from ... {}'.format(model_path))
    keras_model.load_weights(weights_path)


    # =======  Feature importance  =======
    # Compute correlated features subgroups
    col_sets = pfi_utils.get_fea_groups(xvl, th=corr_th, toplot=False)

    if len(col_sets) == 0:
        col_sets = [[c] for c in xdata.columns.tolist()]
    else:
        cols_unq_req = set()  # set of unique cols that were requested
        for col_set in col_sets:  # get the unique cols that were passed in col_sets
            for col in col_set:
                cols_unq_req.add(col)
        cols_unq = set(xdata.columns.tolist())
        cols_other = cols_unq.difference(cols_unq_req)
        col_sets = sorted(col_sets, key=len, reverse=True)  # sort list based on the length of sublists
        col_sets.extend([[c] for c in cols_other])
        col_sets = col_sets    

    # Create df that stores feature importance
    fi_var = pd.DataFrame(index=range(len(col_sets)), columns=['cols', 'imp'])
    fi_score = pd.DataFrame(index=range(len(col_sets)), columns=['cols', 'imp', 'std'])

    # Iter over col sets (col set per node)
    print('Iterate over col sets to compute importance ...')
    for i, col_set in enumerate(col_sets):
        # pred_df = infer_with_col_shuffle_multi(model=keras_model, xdata=xdata, col=col, n_shuffles=n_shuffles)
        pred_df = pfi_utils.shuf_and_pred_multi(model=keras_model, xdata=xdata,
                                                col_set=col_set, n_shuffles=n_shuffles)

        # ## ----------------------------------------------------------------------
        # pred_df = pd.DataFrame(index=range(xdata.shape[0]), columns=range(n_shuffles))
        # for s in range(n_shuffles):
        #     # Execute infer code
        #     # TODO: still need to decide regarding the output ...
        #     preds = shuf_and_pred(model=keras_model, xdata=xdata, col=col)
        #     pred_df.iloc[:, s] = preds
        # ## ----------------------------------------------------------------------

        pred_df.to_csv(os.path.join(tempdir, 'col.' + str('-'.join(col_set)) + '.csv'), sep='\t', index=False)

        fi_var.loc[i, 'cols'] = ','.join(col_set)  # col
        fi_var.loc[i, 'imp'] = pred_df.var(axis=1).mean()

    fi_var['imp'] = fi_var['imp'] / fi_var['imp'].sum()
    fi_var = fi_var.sort_values('imp', ascending=False).reset_index(drop=True)
    fi_var.to_csv(os.path.join(OUTDIR, 'fi_var.csv'), index=False)


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
