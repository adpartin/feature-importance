"""
This code is used to analyze the run-time performance of the PFI alagorithm.
Specifically, the following run-time is recorded as a function of 2 parameters
(n_samples, n_features). Plot: (n_samples, n_features) vs runtime
"""
from __future__ import print_function, division

import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils

APP = 'tc'
DATAPATH = os.path.join(file_path, 'data', f'{APP}_data')
SEED = 0


def run():
    # print(args)
    n_shuffles = 20
    corr_th = 1

    # Create necessary dirs
    OUTDIR = os.path.join(file_path, f'results_aacr_{APP}_cor{corr_th}_runtime')
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    # ==========  RF classifier  ==========
    print('\nLoad TC data ...')

    # ---------- Load data ----------
    data = pd.read_csv(DATAPATH, sep='\t')
    xdata = data.iloc[:, 1:].copy()
    ydata = data.iloc[:, 0].copy()
    features = xdata.columns

    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)
    print('\nxtr.shape', xtr.shape)
    print('xvl.shape', xvl.shape)

    # ---------- Feature importance from RF and PFI ----------
    print('\nCompute PFI ...')
    n_samples = np.linspace(start=int(xvl.shape[0]/4), stop=xvl.shape[0], num=4, dtype=int) 
    n_cols = np.linspace(start=int(xvl.shape[1]/4), stop=xvl.shape[1], num=4, dtype=int)
    print(n_samples)
    print(n_cols)

    tt = pd.DataFrame(index=range(len(n_samples) * len(n_cols)),
                      columns=['n_samples', 'n_cols', 'time (sec)', 'time (min)'])

    t_run = time.time()
    cnt = 0
    for i, s in enumerate(n_samples):
        for j, c in enumerate(n_cols):
            print(f'(n_samples, n_cols): ({s}, {c})')
            xtr_ = xtr.iloc[:, :c]
            xvl_ = xvl.iloc[:s, :c]
            yvl_ = yvl[:s]
            # print('xtr_.shape', xtr_.shape)
            # print('xvl_.shape', xvl_.shape)
            # print('yvl_.shape', yvl_.shape)
            
            rf_model = RandomForestClassifier(n_estimators=150, max_features='sqrt', random_state=SEED)
            rf_model.fit(xtr_, ytr)

            fi_obj = pfi.PFI(model=rf_model, xdata=xvl_, ydata=yvl_, n_shuffles=n_shuffles, outdir=OUTDIR)
            fi_obj.gen_col_sets(th=corr_th, toplot=False, verbose=False)
            
            t0 = time.time()
            fi_obj.compute_pfi(ml_type='c', verbose=False)
            t = time.time()-t0
            tt.loc[cnt, ['n_samples', 'n_cols', 'time (sec)', 'time (min)']] = np.array([s, c, t, t/60])
            cnt += 1

    tt.to_csv(os.path.join(OUTDIR, 'tt.csv'), index=False)
    print(f'\nTotal run time:  {(time.time()-t_run)/60} mins')


def main():
    # args = init_params()
    # run(args)
    run()


if __name__ == '__main__':
    main()
