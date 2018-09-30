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

OUTDIR = os.path.join(file_path, 'results_tc_pfi_aacr')
DATAPATH = os.path.join(file_path, 'data', 'tc_data')
APP = 'tc'
N_SHUFFLES = 20
CORR_THRES = 0.75
EPOCH = 60
BATCH = 32
MAX_COLS = 20
SEED = 0


def create_nn_classifier(n_features, n_classes):
    keras_model = Sequential()
    keras_model.add(Dense(units=32, activation='relu', input_shape=(n_features,)))
    if n_classes == 2:
        keras_model.add(Dense(units=1, activation='sigmoid'))
        keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif n_classes > 2:
        keras_model.add(Dense(units=n_classes, activation='softmax'))
        keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return keras_model


def create_nn_regressor(n_features):
    keras_model = Sequential()
    keras_model.add(Dense(units=10, activation='relu', input_shape=(n_features,)))
    keras_model.add(Dense(units=1, activation=None))
    keras_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return keras_model


def init_params():
    parser = argparse.ArgumentParser(description='Compute feature importance via feature shuffling (supports keras).')
    parser.add_argument('-ns', '--n_shuffles', dest='n_shuffles',
                        default=N_SHUFFLES, type=int,
                        help=f'Number of shuffles (default: {N_SHUFFLES}).')
    parser.add_argument('-th', '--corr_th', dest='corr_th',
                        default=CORR_THRES, type=float,
                        help=f'Correlation threshold for which to group the columns for pfi (default: {CORR_THRES}).')
    parser.add_argument('-e', '--epoch', dest='epoch',
                        default=EPOCH, type=int,
                        help=f'Number of epochs (default: {EPOCH}).')
    parser.add_argument('-b', '--batch', dest='batch',
                        default=EPOCH, type=int,
                        help=f'Batch size (default: {BATCH}).')
    parser.add_argument('-mc', '--max_cols', dest='max_cols',
                        default=MAX_COLS, type=int,
                        help=f'Maxium cols to plot for feature importance (default: {MAX_COLS}).')
    parser.add_argument('-bs', '--bootstrap_cols', dest='bootstrap_cols',
                        default=-1, type=int,
                        help=f'The number of cols to bootsrap from the dataframe.')
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


    # ==========  RF classifier  ==========
    print('\nLoad TC data ...')

    # ---------- Load data ----------
    data = pd.read_csv(DATAPATH, sep='\t')
    xdata = data.iloc[:, 1:].copy()
    ydata = data.iloc[:, 0].copy()

    if args.bootstrap_cols > -1:
        xdata = xdata.sample(n=args.bootstrap_cols, axis=1, random_state=SEED)  # Take a subset of cols
    features = xdata.columns

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])
    print('np.unique(ydata)', np.unique(ydata))

    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)
    xdata = pd.DataFrame(xdata, columns=features)

    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)

    # # k-fold scheme
    # kfolds = 5
    # if kfolds == 1:
    #     skf = StratifiedShuffleSplit(n_splits=kfolds, test_size=0.2, random_state=SEED)
    # else:
    #     skf = StratifiedKFold(n_splits=kfolds, shuffle=False, random_state=SEED)

    # # Run k-fold CV
    # best_model = None
    # best_model_id = 0
    # best_score = 0
    # df_scores = pd.DataFrame(index=range(kfolds), columns=['kfold', 'f1_micro', 'f1_macro'])

    # for f, (train_idx, val_idx) in enumerate(skf.split(xdata, ydata)):
    #     print(f'\nFold {f + 1}/{kfolds} ...\n')

    #     print('train_idx', train_idx)
    #     print('val_idx', val_idx)

    #     # Split data
    #     xtr, xvl = xdata[train_idx], xdata[val_idx]
    #     ytr, yvl = ydata[train_idx], ydata[val_idx]

    #     rf_model = RandomForestClassifier(n_estimators=150, max_features='sqrt', random_state=SEED)  # min_samples_split=3,
    #     rf_model.fit(xtr, ytr)
    #     score = rf_model.score(xvl, yvl)
    #     print(f'Prediction score (mean accuracy): {score:.4f}')

    #     yvl_preds = rf_model.predict(xvl)
    #     print('true', yvl[:7])
    #     print('pred', yvl_preds[:7])
    #     print(f'f1_score micro: {f1_score(y_true=yvl, y_pred=yvl_preds, average='micro'):.3f}')
    #     print(f'f1_score macro: {f1_score(y_true=yvl, y_pred=yvl_preds, average='macro'):.3f}')
    #     tmp_df = pd.DataFrame({'yvl': yvl, 'yvl_preds': yvl_preds})
    #     tmp_df.to_csv(os.path.join(OUTDIR, f'preds_cv_{f}.csv'), index=False)

    #     # Plot feature importance
    #     indices, fig = utils.plot_rf_fi(rf_model, n_features_toplot=15, title='FI RF Classifier')
    #     fi = utils.get_rf_fi(rf_model)
    #     fi.to_csv(os.path.join(OUTDIR, 'rf_classifier_fi.csv'), index=False)
    #     fig.savefig(os.path.join(OUTDIR, 'rf_classifier_fi.png'), bbox_inches='tight')

    #     # Compute scores
    #     df_scores.loc[f, 'kfold'] = f + 1
    #     df_scores.loc[f, 'f1_micro'] = f1_score(y_true=yvl, y_pred=yvl_preds, average='micro')
    #     df_scores.loc[f, 'f1_macro'] = f1_score(y_true=yvl, y_pred=yvl_preds, average='macro')

    #     # Save best model
    #     ## if val_scores.iloc[f, 0] < best_score:
    #     if best_score < df_scores.loc[f, 'f1_micro']:
    #         best_score = df_scores.loc[f, 'f1_micro']
    #         best_model = rf_model
    #         best_model_id = f

    # print(df_scores)
    # model = best_model

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
    # cor = utils.compute_cor_mat(xvl, zero_diag=True, decimals=5)
    # fig = utils.plot_cor_heatmap(cor)
    # fig.savefig(os.path.join(OUTDIR, f'{APP}_feature_corr.png'), bbox_inches='tight')

    # ---------- Feature importance from RF and PFI ----------
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, max_cols=max_cols, title='RF Classifier (FI using MDI)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_fi.png'), bbox_inches='tight')

    # PFI
    t0 = time.time()
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c')
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(max_cols=max_cols, title='RF Classifier (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_var.png'), bbox_inches='tight')
    fig = fi_obj.plot_score_fi(max_cols=max_cols, title='RF Classifier (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_score.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name=f'{APP}')


    # # ==========  NN classifier  ==========
    # # print('\nLoading classification dataset...')
    # #
    # # # Load data
    # # data = pd.read_csv(DATAPATH, sep='\t')
    # # xdata = data.iloc[:, 1:].values
    # # ydata = data.iloc[:, 0].values
    # #
    # # print('data.shape', data.shape)
    # # print(data.iloc[:3, :4])
    # # print('np.unique(ydata)', np.unique(ydata))
    #
    # n_classes = len(np.unique(ydata))
    # # ydata = keras.utils.to_categorical(ydata, num_classes=n_classes)
    # ytr = keras.utils.to_categorical(ytr, num_classes=n_classes)
    # yvl = keras.utils.to_categorical(yvl, num_classes=n_classes)
    #
    # # Scale data
    # # scaler = StandardScaler()
    # # xdata = scaler.fit_transform(xdata)
    #
    # # # Drop cols (high correlation)
    # # xdata, corr, cols_dropped = drop_high_correlation_cols(pd.DataFrame(xdata), thres_corr=thres_corr, verbose=True)
    # # print('xdata.shape', xdata.shape)
    # # print('xdata.columns', xdata.columns.tolist())
    #
    # # Split data
    # # xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)
    #
    # print('\nTraining NN Classifier...')
    # keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    # keras_model.fit(xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    # score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    # print('Prediction score (val loss): {:.4f}'.format(score))
    #
    # yvl_preds = keras_model.predict(xvl)
    # print('true', np.argmax(yvl[:10], axis=1))
    # print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    # print('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    # print('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))
    #
    # # Permutation feature importance using var
    # t0 = time.time()
    # # fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    # fi_var = fi_obj.compute_feature_importance_var()
    # fi_var.to_csv(os.path.join(OUTDIR, 'nn_classifier_pfi_var.csv'), index=False)
    # fig = fi_obj.plot_fi(title='NN Classifier (PFI var)', ylabel='Importance (normalized)')
    # fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_var.png'), bbox_inches='tight')
    # print('Total running time NN: {:.3f} min'.format((time.time() - t0) / 60))
    #
    # # # Permutation feature importance using score
    # # t0 = time.time()
    # # # fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # # fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    # # fi_score = fi_obj.compute_feature_importance_score(ml_type='c', verbose=True)
    # # fi_score.to_csv(os.path.join(OUTDIR, 'nn_classifier_pfi_score.csv'), index=False)
    # # fig = fi_obj.plot_fi(title='NN Classifier (PFI score: f1-score)', ylabel='Importance (score decrease)')
    # # fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_score.png'), bbox_inches='tight')
    # # print('Total running time NN: {:.3f} min'.format((time.time() - t0) / 60))


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
