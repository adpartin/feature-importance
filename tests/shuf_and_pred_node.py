from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# MODELDIR = os.path.join(file_path, 'keras_model')
TEMPDIR = os.path.join(file_path, 'tempdir')
OUTDIR = os.path.join(file_path, 'test_results_node')
SEED = 0


def init_params():
    parser = argparse.ArgumentParser(description='Shuffle and predict per node.')
    parser.add_argument('-d', '--datapath', dest='datapath',
                        default=DATAPATH,
                        help='full path to the data')
    parser.add_argument('-m', '--modeldir', dest='modeldir',
                        default=MODELDIR,
                        help='directory where the trained DL model (model and weights) is stored')
    parser.add_argument('-t', '--tempdir', dest='tempdir',
                        default=TEMPDIR,
                        help='directory to store the (temp) predictions for the current col shuffle')
    parser.add_argument('-ns', '--n_shuffles', dest='n_shuffles',
                        default=20, type=int,
                        help='number of shuffles')
    parser.add_argument('-c', '--col_set', dest='col_set',
                        type=int,
                        help='column id to shuffle')
    return parser.parse_args()


def run(args):
    print(args)
    datapath = args.datapath
    modeldir = args.modeldir
    tempdir = args.tempdir
    n_shuffles = args.n_shuffles
    # col_set = args.col_set
    col_set = ['C', 'F']

    # Create necessary dirs
    utils.make_dir(tempdir)  #  os.makedirs(tempdir, exist_ok=True)

    # Load data
    data = pd.read_csv(datapath, sep=None)
    # ydata = data.iloc[:, 0].values
    xdata = data.iloc[:, 1:].values
    features = xdata.columns

    # # Scale data
    # scaler = StandardScaler()
    # xdata = scaler.fit_transform(xdata)
    # xdata = pd.DataFrame(xdata, columns=features)

    # Load trained keras model
    # (no need to compile the model for inference)
    model_name = 'keras_model.json'
    weights_name = 'keras_weights.h5'

    model_path = os.path.join(modeldir, model_name)
    weights_path = os.path.join(modeldir, weights_name)

    print('\nLoading model from ... {}'.format(model_path))
    with open(model_path, 'r') as json_file:
        model = json_file.read()
    keras_model = model_from_json(model)
    keras_model.name = 'trained_model'

    # Load weights into new model
    print('Loading weights from ... {}\n'.format(weights_path))
    keras_model.load_weights(weights_path)


    # Shuffle and predict
    # pred_df = infer_with_col_shuffle_multi(model=keras_model, xdata=xdata, col=col, n_shuffles=n_shuffles)
    pred_df = pfi_utils.shuf_and_pred_multi(model=keras_model, xdata=xdata,
                                            col_set=col_set, n_shuffles=n_shuffles)

    # ## ----------------------------------------------------------------------
    # pred_df = pd.DataFrame(index=range(len(xdata)), columns=range(n_shuffles))

    # for s in range(n_shuffles):
    #     # Execute infer code
    #     preds = infer_with_col_shuffle(model, xdata, col)
    #     pred_df.iloc[:, s] = preds
    # ## ----------------------------------------------------------------------


    # Write out results
    # pred_df.to_csv(os.path.join(tempdir, 'col.' + str(col) + '.csv'), sep='\t', index=False)
    pred_df.to_csv(os.path.join(tempdir, 'col.' + str('-'.join(col_set)) + '.csv'), sep='\t', index=False)


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
