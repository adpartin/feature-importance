"""
This code:
* Loads classification and regression datasets which may contain highly
  correlated (the data are generated using create_dataset.py).
* Trains RandomForestClassifier and RandomForestRegressor.
* Computes feature importance using MDI (available in sklearn) and PFI.
* Generates and saves feature importance plots.
"""
from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils

DATAPATH_CLASSIFICATION = os.path.join(file_path, 'data', 'data_classification_corr')
DATAPATH_REGRESSION = os.path.join(file_path, 'data', 'data_regression_corr')
OUTDIR = os.path.join(file_path, 'test_results')
# DATAPATH_CLASSIFICATION = os.path.join(file_path, 'data', 'data_classification')
# DATAPATH_REGRESSION = os.path.join(file_path, 'data', 'data_regression')
# OUTDIR = os.path.join(file_path, 'test_results_col_sets')
N_SHUFFLES = 20
CORR_THRES = 0.74
SEED = 0


def init_params():
    parser = argparse.ArgumentParser(description='Compute permutation feature importance (with col_set support).')
    parser.add_argument('-ns', '--n_shuffles', dest='n_shuffles',
                        default=N_SHUFFLES, type=int,
                        help='number of shuffles (default: {})'.format(N_SHUFFLES))
    parser.add_argument('-th', '--corr_th', dest='corr_th',
                        default=CORR_THRES, type=float,
                        help='correlation threshold for which to group the columns for pfi (default: {})'.format(CORR_THRES))
    return parser.parse_args()


def run(args):
    print(args)
    n_shuffles = args.n_shuffles
    corr_th = args.corr_th

    # Create necessary dirs
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    
    # ==========  RF classifier  ==========
    print('\nLoad classification data ...')

    # ---------- Load data ----------
    data = pd.read_csv(DATAPATH_CLASSIFICATION, sep='\t')
    xdata = data.iloc[:, 1:].copy()
    ydata = data.iloc[:, 0].copy()
    features = xdata.columns

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])
    print('np.unique(ydata)', np.unique(ydata))

    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)

    # ---------- Train classifier ----------
    print('\nTrain RF Classifier ...')
    rf_model = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=SEED)
    rf_model.fit(xtr, ytr)
    print(f'Prediction score (mean accuracy): {rf_model.score(xvl, yvl):.4f}')

    yvl_preds = rf_model.predict(xvl)
    print('true', yvl[:10].values)
    print('pred', yvl_preds[:10])
    print('f1_score micro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='micro')))
    print('f1_score macro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='macro')))

    # Compute corr matrix
    cor = utils.compute_cor_mat(xvl, zero_diag=True, decimals=5)
    fig = utils.plot_cor_heatmap(cor)
    fig.savefig(os.path.join(OUTDIR, 'feature_corr_classification.png'), bbox_inches='tight')

    # ---------- Feature importance from RF and PFI ----------
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, title='RF Classifier (FI using MDI)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_fi.png'), bbox_inches='tight')

    # PFI
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(title='RF Classifier (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_var.png'), bbox_inches='tight')
    fig = fi_obj.plot_score_fi(title='RF Classifier (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_score.png'), bbox_inches='tight')

    # Save PFI tables
    fi_obj.dump_fi(path=OUTDIR, name='classifier')
    fi_obj.dump_col_sets(path=OUTDIR, name='classifier')


    # ==========  RF regressor  ==========
    print('\nLoad regression data ...')

    # ---------- Load data ----------
    data = pd.read_csv(DATAPATH_REGRESSION, sep='\t')
    xdata = data.iloc[:, 1:].copy()
    ydata = data.iloc[:, 0].copy()
    features = xdata.columns

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])

    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True)

    # ---------- Train regressor ----------
    print('\nTrain RF Regressor ...')
    rf_model = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=SEED)
    rf_model.fit(xtr, ytr)
    score = rf_model.score(xvl, yvl)
    print(f'Prediction score (r_square): {score:.4f}')

    # Compute corr matrix
    cor = utils.compute_cor_mat(xvl, zero_diag=True, decimals=5)
    fig = utils.plot_cor_heatmap(cor)
    fig.savefig(os.path.join(OUTDIR, 'feature_corr_regression.png'), bbox_inches='tight')

    # ---------- Feature importance from RF and PFI ----------
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, title='RF Regressor (FI using MDI)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_fi.png'), bbox_inches='tight')

    # PFI
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='r')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(title='RF Regressor (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_pfi_var.png'), bbox_inches='tight')
    fig = fi_obj.plot_score_fi(title='RF Regressor (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_pfi_score.png'), bbox_inches='tight')

    # Save PFI tables
    fi_obj.dump_fi(path=OUTDIR, name='regressor')
    fi_obj.dump_col_sets(path=OUTDIR, name='regressor')


    # # Create validation set with only important features
    # xtr_imp = xtr[:, indices[:args.n_important]]
    # xvl_imp = xvl[:, indices[:args.n_important]]
    #
    # # Create validation set with only non-important (dummy) features
    # xtr_dum = xtr[:, indices[args.n_important:]]
    # xvl_dum = xvl[:, indices[args.n_important:]]
    #
    # # Build RF using only important features
    # rf_model_imp = RandomForestClassifier(n_estimators=250, random_state=SEED)
    # rf_model_imp.fit(xtr_imp, ytr)
    # score = rf_model_imp.score(xvl_imp, yvl)
    # print('Prediction score (using only important features): {:.4f}'.format(score))
    #
    # # Build a forest using the non-important features
    # rf_model_dum = RandomForestClassifier(n_estimators=250, random_state=SEED)
    # rf_model_dum.fit(xtr_dum, ytr)
    # score = rf_model_dum.score(xvl_dum, yvl)
    # print('Prediction score (using only non-important features): {:.4f}'.format(score))


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()