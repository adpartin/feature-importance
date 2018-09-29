from __future__ import print_function, division

import os
import sys
import string
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import *

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

OUTDIR = os.path.join(file_path, 'data')
N_SAMPLES = 2000
N_CLASSES = 2
N_FEATURES = 10
N_INFORMATIVE = 5
N_REDUNDANT = 2
N_REPEATED = 0
SEED = 0


def init_params():
    parser = argparse.ArgumentParser(description='Generates classification and regression datasets.')
    parser.add_argument('-s', '--n_samples', dest='n_samples',
                        default=N_SAMPLES, type=int,
                        help='number of samples/observations/rows (default: {})'.format(N_SAMPLES))
    parser.add_argument('-c', '--n_classes', dest='n_classes',
                        default=N_CLASSES, type=int,
                        help='number of target classes (for classification) (default: {})'.format(N_CLASSES))
    parser.add_argument('-f', '--n_features', dest='n_features',
                        default=N_FEATURES, type=int,
                        help='number of features/variables/columns (default: {})'.format(N_FEATURES))
    parser.add_argument('-i', '--n_informative', dest='n_informative',
                        default=N_INFORMATIVE, type=int,
                        help='number of important/informative features (default: {})'.format(N_INFORMATIVE))
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
    print('\nGenerate classification dataset...')
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
    df = pd.concat([ydata, xdata], axis=1)

    print('df.shape', df.shape)
    print(df.iloc[:3, :4])

    if (N_REDUNDANT == 0) and (N_REPEATED == 0):
        df.to_csv(os.path.join(OUTDIR, 'data_classification'), sep='\t', float_format=np.float16, index=False)
    else:
        df.to_csv(os.path.join(OUTDIR, 'data_classification_corr'), sep='\t', float_format=np.float16, index=False)
        

    # Build regression dataset
    print('\nGenerate regression dataset...')
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
                                   shuffle=False)

    ydata = pd.DataFrame(ydata).rename(columns={0: 'y'})
    # xdata = pd.DataFrame(xdata)
    xdata = pd.DataFrame(xdata, columns=[c for c in string.ascii_uppercase[:xdata.shape[1]]])
    df = pd.concat([ydata, xdata], axis=1)

    print('df.shape', df.shape)
    print(df.iloc[:3, :4])

    if (N_REDUNDANT == 0) and (N_REPEATED == 0):
        df.to_csv(os.path.join(OUTDIR, 'data_regression'), sep='\t', float_format=np.float16, index=False)
    else:
        df.to_csv(os.path.join(OUTDIR, 'data_regression_corr'), sep='\t', float_format=np.float16, index=False)


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
