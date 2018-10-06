from __future__ import print_function, division

import os
import sys
import string
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import *

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

OUTDIR = os.path.join(file_path, 'data')
# N_SAMPLES = 2000
N_CLASSES = 3
N_SAMPLES_PER_CLASS = 100
TEST_SIZE = 0.2
N_SAMPLES = int(N_SAMPLES_PER_CLASS * N_CLASSES / TEST_SIZE)
N_FEATURES = 7
N_INFORMATIVE = 4
N_REDUNDANT = 0
N_REPEATED = 0
SEED = 0


def init_params():
    parser = argparse.ArgumentParser(description='Generates classification and regression datasets.')
    parser.add_argument('-s', '--n_samples', dest='n_samples',
                        default=N_SAMPLES, type=int,
                        help=f'number of samples/observations/rows (default: {N_SAMPLES})')
    parser.add_argument('-c', '--n_classes', dest='n_classes',
                        default=N_CLASSES, type=int,
                        help=f'number of target classes (for classification) (default: {N_CLASSES})')
    parser.add_argument('-f', '--n_features', dest='n_features',
                        default=N_FEATURES, type=int,
                        help=f'number of features/variables/columns (default: {N_FEATURES})')
    parser.add_argument('-i', '--n_informative', dest='n_informative',
                        default=N_INFORMATIVE, type=int,
                        help=f'number of important/informative features (default: {N_INFORMATIVE})')
    return parser.parse_args()


def run(args):
    print(args)
    n_samples = args.n_samples
    n_classes = args.n_classes
    n_features = args.n_features
    n_informative = args.n_informative

    # Create necessary dirs
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    # Build classification dataset
    print('\n======= Generate classification data =======')
    xdata, ydata = make_classification(n_samples=n_samples,
                                       n_classes=n_classes,
                                       n_features=n_features,
                                       n_informative=n_informative,
                                       n_redundant=N_REDUNDANT,  # features generated as random linear combinations of the informative features
                                       n_repeated=N_REPEATED,  # duplicated features, drawn randomly from the informative and the redundant features
                                       shift=None,
                                       scale=None,
                                       random_state=SEED,
                                       shuffle=False)

    ydata = pd.DataFrame(ydata).rename(columns={0: 'y'})
    # xdata = pd.DataFrame(xdata)
    xdata = pd.DataFrame(xdata, columns=[c for c in string.ascii_uppercase[:xdata.shape[1]]])
    data = pd.concat([ydata, xdata], axis=1)
    data = data.sample(data.shape[0], axis=0, replace=False, random_state=SEED)

    features = xdata.columns
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=ydata)
    data_train = pd.concat([ytr, xtr], axis=1)
    data_val = pd.concat([yvl, xvl], axis=1)

    # Sort val data by class label
    data_val = data_val.sort_values('y', ascending=True).reset_index(drop=True)

    print('data.shape      ', data.shape)
    print('data_train.shape', data_train.shape)
    print('data_val.shape  ', data_val.shape)
    print(data.iloc[:3, :4])


    if (N_REDUNDANT == 0) and (N_REPEATED == 0):
        # data.to_csv(os.path.join(OUTDIR, 'data_classification'), sep='\t', float_format=np.float16, index=False)
        data_train.to_csv(os.path.join(OUTDIR, 'data_classification_train'), sep='\t', float_format=np.float16, index=False)
        data_val.to_csv(os.path.join(OUTDIR, 'data_classification_val'), sep='\t', float_format=np.float16, index=False)
    else:
        # data.to_csv(os.path.join(OUTDIR, 'data_classification_corr'), sep='\t', float_format=np.float16, index=False)
        data_train.to_csv(os.path.join(OUTDIR, 'data_classification_corr_train'), sep='\t', float_format=np.float16, index=False)
        data_val.to_csv(os.path.join(OUTDIR, 'data_classification_corr_val'), sep='\t', float_format=np.float16, index=False)


    # Build regression dataset
    print('\n======= Generate regression data =======')
    xdata, ydata = make_regression(n_samples=n_samples,
                                   n_targets=1,
                                   n_features=n_features,
                                   n_informative=n_informative,
                                   bias=0.0,
                                   effective_rank=None,
                                   tail_strength=0.5,
                                   noise=0.0,
                                   coef=False,
                                   random_state=SEED,
                                   shuffle=True)

    ydata = pd.DataFrame(ydata).rename(columns={0: 'y'})
    # xdata = pd.DataFrame(xdata)
    xdata = pd.DataFrame(xdata, columns=[c for c in string.ascii_uppercase[:xdata.shape[1]]])
    data = pd.concat([ydata, xdata], axis=1)
    data = data.sample(data.shape[0], replace=False, random_state=SEED)

    features = xdata.columns
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True)
    data_train = pd.concat([ytr, xtr], axis=1)
    data_val = pd.concat([yvl, xvl], axis=1)

    print('data.shape      ', data.shape)
    print('data_train.shape', data_train.shape)
    print('data_val.shape  ', data_val.shape)
    print(data.iloc[:3, :4])

    if (N_REDUNDANT == 0) and (N_REPEATED == 0):
        # data.to_csv(os.path.join(OUTDIR, 'data_regression'), sep='\t', float_format=np.float16, index=False)
        data_train.to_csv(os.path.join(OUTDIR, 'data_regression_train'), sep='\t', float_format=np.float16, index=False)
        data_val.to_csv(os.path.join(OUTDIR, 'data_regression_val'), sep='\t', float_format=np.float16, index=False)        
    else:
        # data.to_csv(os.path.join(OUTDIR, 'data_regression_corr'), sep='\t', float_format=np.float16, index=False)
        data_train.to_csv(os.path.join(OUTDIR, 'data_regression_corr_train'), sep='\t', float_format=np.float16, index=False)
        data_val.to_csv(os.path.join(OUTDIR, 'data_regression_corr_val'), sep='\t', float_format=np.float16, index=False)        


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
