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
import time
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
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

import warnings
warnings.filterwarnings('ignore')

# DATAPATH_CLASSIFICATION = os.path.join(file_path, 'data', 'data_classification')
# DATAPATH_REGRESSION = os.path.join(file_path, 'data', 'data_regression')

DATAPATH_CLASSIFICATION_TRAIN = os.path.join(file_path, 'data', 'data_classification_train')
DATAPATH_CLASSIFICATION_VAL = os.path.join(file_path, 'data', 'data_classification_val')
DATAPATH_REGRESSION_TRAIN = os.path.join(file_path, 'data', 'data_regression_train')
DATAPATH_REGRESSION_VAL = os.path.join(file_path, 'data', 'data_regression_val')

# DATAPATH_CLASSIFICATION = os.path.join(file_path, 'data', 'data_classification_corr')
# DATAPATH_REGRESSION = os.path.join(file_path, 'data', 'data_regression_corr')
OUTDIR = os.path.join(file_path, 'results_test_pfi')
N_SHUFFLES = 20
CORR_THRES = 0.74
EPOCH = 60
BATCH = 32
MAX_COLS = 20
SEED = 0


def create_nn_classifier(n_features, n_classes):
    keras_model = Sequential()
    keras_model.add(Dense(units=32, activation='relu', input_shape=(n_features,)))
    # if n_classes == 2:
    #     keras_model.add(Dense(units=1, activation='sigmoid'))
    #     keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # elif n_classes > 2:
    keras_model.add(Dense(units=n_classes, activation='softmax'))
    keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return keras_model


def create_nn_regressor(n_features):
    keras_model = Sequential()
    keras_model.add(Dense(units=32, activation='relu', input_shape=(n_features,)))
    keras_model.add(Dense(units=1, activation=None))
    keras_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return keras_model


def init_params():
    parser = argparse.ArgumentParser(description='Compute permutation feature importance (with col_set support).')
    parser.add_argument('-ns', '--n_shuffles', dest='n_shuffles',
                        default=N_SHUFFLES, type=int,
                        help=f'number of shuffles (default: {N_SHUFFLES})')
    parser.add_argument('-th', '--corr_th', dest='corr_th',
                        default=CORR_THRES, type=float,
                        help=f'correlation threshold for which to group the columns for pfi (default: {CORR_THRES})')
    parser.add_argument('-e', '--epoch', dest='epoch',
                        default=EPOCH, type=int,
                        help=f'number of epochs (default: {EPOCH})')
    parser.add_argument('-b', '--batch', dest='batch',
                        default=EPOCH, type=int,
                        help=f'batch size (default: {BATCH})')
    parser.add_argument('-mc', '--max_cols', dest='max_cols',
                        default=MAX_COLS, type=int,
                        help=f'batch size (default: {MAX_COLS})')                         
    return parser.parse_args()


def run(args):
    print(args)
    n_shuffles = args.n_shuffles
    corr_th = args.corr_th
    epoch = args.epoch
    batch = args.batch
    max_cols = args.max_cols

    # Create necessary dirs
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)


    # ==========  Load classification data  ==========
    print('\n======== Load classification data ========')
    data_train = pd.read_csv(DATAPATH_CLASSIFICATION_TRAIN, sep='\t')
    data_val   = pd.read_csv(DATAPATH_CLASSIFICATION_VAL, sep='\t')
    print('data_train.shape', data_train.shape)
    print('data_val.shape  ', data_val.shape)
    print(f'\ndata_train:\n{data_train.iloc[:3, :4]}')
    print(f'\ndata_val:\n{data_val.iloc[:3, :4]}')


    # ==========  RF classifier  ==========
    xtr = data_train.iloc[:, 1:].copy()
    ytr = data_train.iloc[:, 0].copy()
    xvl = data_val.iloc[:, 1:].copy()
    yvl = data_val.iloc[:, 0].copy()
    features = xtr.columns
    print(f'\nnp.unique(ytr): {np.unique(ytr)}')

    # Compute corr matrix
    cor = utils.compute_cor_mat(xvl, zero_diag=True, decimals=5)
    fig = utils.plot_cor_heatmap(cor)
    fig.savefig(os.path.join(OUTDIR, 'feature_corr_classification.png'), bbox_inches='tight')

    # ---------- Train classifier ----------
    print('\n------- Train RF Classifier -------')
    rf_model = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=SEED)
    rf_model.fit(xtr, ytr)
    print(f'Prediction score (mean accuracy): {rf_model.score(xvl, yvl):.4f}')

    yvl_preds = rf_model.predict(xvl)
    print('true', yvl[:5].values)
    print('pred', yvl_preds[:5])
    print('f1_score micro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='micro')))
    print('f1_score macro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='macro')))

    yvl_preds_p = rf_model.predict_proba(xvl)
    print(f'yvl_preds_p:\n{yvl_preds_p[:5]}')

    utils.plot_confusion_matrix(y_true=yvl, y_pred=yvl_preds, labels=yvl.unique(),
                                title=f'RF Classifier (Confusion)', savefig=True,
                                img_name=os.path.join(OUTDIR, 'rf_classifier_confusion.png'))

    # ---------- MDI and PFI from RF ----------
    print('\n------- MDI and PFI from RF classifier -------')
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, title='RF Classifier (FI using MDI)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_fi.png'), bbox_inches='tight') 

    # PFI
    t0 = time.time()
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles, outdir=OUTDIR)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c', verbose=False)
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(title='RF Classifier (PFI var)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_var.png'), bbox_inches='tight')

    fig = fi_obj.plot_score_fi(title='RF Classifier (PFI MDA: f1-score)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_score.png'), bbox_inches='tight')
    
    fig = fi_obj.plot_score_fi_p(title='RF Classifier (PFI MDA: p-score)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_score_p.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name='rf_classifier')


    # ==========  NN classifier  ==========
    # print('\nLoad classification data ...')

     # ---------- Load data ----------
    # data = pd.read_csv(DATAPATH_CLASSIFICATION, sep='\t')
    # xdata = data.iloc[:, 1:].copy()
    # ydata = data.iloc[:, 0].copy()
    # features = xdata.columns

    # print('data.shape', data.shape)
    # print(data.iloc[:3, :4])

    # print('\nxdata.shape', xdata.shape)
    # print('np.unique(ydata)', np.unique(ydata))

    # n_classes = len(np.unique(ydata))
    # ydata = keras.utils.to_categorical(ydata, num_classes=n_classes)

    # scaler = StandardScaler()
    # xdata = scaler.fit_transform(xdata)
    # xdata = pd.DataFrame(xdata, columns=features)

    # xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)
    
    # n_classes = len(np.unique(ydata))
    # ytr = keras.utils.to_categorical(ytr, num_classes=n_classes)
    # yvl = keras.utils.to_categorical(yvl, num_classes=n_classes)

    # print('\nTrain NN Classifier ...')
    # keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    # history = keras_model.fit(xtr, ytr, epochs=epoch, batch_size=batch, verbose=0)
    # score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    # print(f'Prediction score (val loss): {score:.4f}')

    # yvl_preds = keras_model.predict(xvl)
    # print('true', np.argmax(yvl[:10], axis=1))
    # print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    # print('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    # print('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))

    # # ---------- Feature importance from RF and PFI ----------
    # # PFI
    # print('\nCompute PFI (NN classifier) ...')
    # t0 = time.time()
    # fi_obj = pfi.PFI(model=keras_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles)
    # fi_obj.gen_col_sets(th=corr_th, toplot=False)
    # fi_obj.compute_pfi(ml_type='c', verbose=False)
    # print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # # Plot and save PFI
    # fig = fi_obj.plot_var_fi(title='NN Classifier (PFI var)', ylabel='Importance (relative)')
    # fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_var.png'), bbox_inches='tight')
    # fig = fi_obj.plot_score_fi(title='NN Classifier (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    # fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_score.png'), bbox_inches='tight')

    # # Dump resutls
    # fi_obj.dump(path=OUTDIR, name='nn_classifier')


    # ==========  Load regression data  ==========
    print('\n======== Load regression data ========')

    data_train = pd.read_csv(DATAPATH_REGRESSION_TRAIN, sep='\t')
    data_val   = pd.read_csv(DATAPATH_REGRESSION_VAL, sep='\t')
    print('data_train.shape', data_train.shape)
    print('data_val.shape  ', data_val.shape)
    print(f'\ndata_train:\n{data_train.iloc[:3, :4]}')
    print(f'\ndata_val:\n{data_val.iloc[:3, :4]}')


    # ==========  RF regressor  ==========
    xtr = data_train.iloc[:, 1:].copy()
    ytr = data_train.iloc[:, 0].copy()
    xvl = data_val.iloc[:, 1:].copy()
    yvl = data_val.iloc[:, 0].copy()
    features = xtr.columns
    print(f'\nnp.unique(ytr): {np.unique(ytr)}')

    # Compute corr matrix
    cor = utils.compute_cor_mat(xvl, zero_diag=True, decimals=5)
    fig = utils.plot_cor_heatmap(cor)
    fig.savefig(os.path.join(OUTDIR, 'feature_corr_regression.png'), bbox_inches='tight')

    # ---------- Train regressor ----------
    print('\n------- Train RF Regressor -------')
    rf_model = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features='sqrt', random_state=SEED)
    rf_model.fit(xtr, ytr)
    score = rf_model.score(xvl, yvl)
    print(f'Prediction score (r_square): {score:.4f}')

    # ---------- Feature importance from RF and PFI ----------
    print('\n------- MDI and PFI from RF regressor -------')
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, title='RF Regressor (FI using MDI)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_fi.png'), bbox_inches='tight')

    # PFI
    t0 = time.time()    
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles, outdir=OUTDIR)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='r', verbose=False)
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(title='RF Regressor (PFI var)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_pfi_var.png'), bbox_inches='tight')
    fig = fi_obj.plot_score_fi(title='RF Regressor (PFI MDA: f1-score)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_pfi_score.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name='rf_regressor')


    # # ==========  NN classifier  ==========
    # print('\nLoad classification data ...')

    #  # ---------- Load data ----------
    # data = pd.read_csv(DATAPATH_CLASSIFICATION, sep='\t')
    # xdata = data.iloc[:, 1:].copy()
    # ydata = data.iloc[:, 0].copy()
    # features = xdata.columns

    # print('data.shape', data.shape)
    # print(data.iloc[:3, :4])

    # print('\nxdata.shape', xdata.shape)
    # print('np.unique(ydata)', np.unique(ydata))

    # n_classes = len(np.unique(ydata))
    # ydata = keras.utils.to_categorical(ydata, num_classes=n_classes)

    # scaler = StandardScaler()
    # xdata = scaler.fit_transform(xdata)
    # xdata = pd.DataFrame(xdata, columns=features)

    # xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)
 
    # print('\nTrain NN Classifier ...')
    # keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    # history = keras_model.fit(xtr, ytr, epochs=epoch, batch_size=batch, verbose=0)
    # score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    # print(f'Prediction score (val loss): {score:.4f}')

    # yvl_preds = keras_model.predict(xvl)
    # print('true', np.argmax(yvl[:10], axis=1))
    # print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    # print('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    # print('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))

    # # ---------- Feature importance from RF and PFI ----------
    # # PFI
    # print('\nCompute PFI (NN classifier) ...')
    # t0 = time.time()
    # fi_obj = pfi.PFI(model=keras_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles)
    # fi_obj.gen_col_sets(th=corr_th, toplot=False)
    # fi_obj.compute_pfi(ml_type='c', verbose=False)
    # print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # # Plot and save PFI
    # fig = fi_obj.plot_var_fi(title='NN Classifier (PFI var)', ylabel='Importance (relative)')
    # fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_var.png'), bbox_inches='tight')
    # fig = fi_obj.plot_score_fi(title='NN Classifier (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    # fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_score.png'), bbox_inches='tight')

    # # Dump resutls
    # fi_obj.dump(path=OUTDIR, name='nn_classifier')


    # ==========  NN regressor  ==========
    # print('\nLoad regression data ...')


    # # Create validation set with only important features
    # xtr_imp = xtr[:, indices[:args.n_important]]
    # xvl_imp = xvl[:, indices[:args.n_important]]
    #
    # # Create validation set with only non-important (dummy) features
    # xtr_dum = xtr[:, indices[args.n_important:]]
    # xvl_dum = xvl[:, indices[args.n_important:]]
    #
    # # Build RF using only important features
    # rf_model_imp = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features='sqrt', random_state=SEED)
    # rf_model_imp.fit(xtr_imp, ytr)
    # score = rf_model_imp.score(xvl_imp, yvl)
    # print('Prediction score (using only important features): {:.4f}'.format(score))
    #
    # # Build a forest using the non-important features
    # rf_model_dum = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features='sqrt', random_state=SEED)
    # rf_model_dum.fit(xtr_dum, ytr)
    # score = rf_model_dum.score(xvl_dum, yvl)
    # print('Prediction score (using only non-important features): {:.4f}'.format(score))


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()