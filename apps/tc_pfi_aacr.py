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
import pdb
import logging
import argparse

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

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

APP = 'P1B1'
# DATAPATH = os.path.join(file_path, 'data', f'{APP}_data')
# DATAPATH_TR = os.path.join(file_path, 'data', f'{APP}_data_train')
# DATAPATH_VL = os.path.join(file_path, 'data', f'{APP}_data_val')

# DATAPATH_TR = os.path.join(file_path, 'data', f'{APP}_data_train_raw')
# DATAPATH_VL = os.path.join(file_path, 'data', f'{APP}_data_val_raw')
# YENC_PATH = os.path.join(file_path, 'data', f'{APP}_y_enc')

# Benchmark TC data for FI (by FF)
DATAPATH_TR = os.path.join(file_path, 'data', f'P1B1.dev.train.lincs.ap')
DATAPATH_VL = os.path.join(file_path, 'data', f'P1B1.dev.test.lincs.ap')
YENC_PATH = os.path.join(file_path, 'data', f'P1B1.y.enc.ap')

N_SHUFFLES = 20
CORR_THRES = 1.0
EPOCH = 200
BATCH = 128
MAX_COLS_PLOT = 25  # default number of cols to display for feature importance
SEED = 0


def set_logger(filename='main_logfile.log'):
    """
    https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
    """
    # TODO: finish this!
    # handlers = [logging.FileHandler(filename),
    #             logging.StreamHandler()]
    # logging.basicConfig(level=logging.INFO,
    #                     format="[%(asctime)s %(process)d] %(message)s",
    #                     datefmt="%Y-%m-%d %H:%M:%S",
    #                     filename=filename,
    #                     handlers=handlers)

    # # create a stream handler
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create a file handler
    fh = logging.FileHandler(filename=filename)
    fh.setLevel(logging.INFO)

    # create a logging format
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    return logger


def create_nn_classifier(n_features, n_classes):
    # https://sebastianraschka.com/faq/docs/dropout-activation.html
    dense_units = 128
    keras_model = Sequential()
    keras_model.add(Dense(units=dense_units, activation='relu', input_shape=(n_features,)))
    keras_model.add(Dense(units=dense_units, activation='relu'))
    keras_model.add(Dense(units=dense_units, activation='relu'))
    keras_model.add(Dense(units=dense_units, activation='relu'))
    keras_model.add(Dense(units=dense_units, activation='relu'))
    # if n_classes == 2:
    #     keras_model.add(Dense(units=1, activation='sigmoid'))
    #     keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # elif n_classes > 2:
    
    # keras_model.add(Dense(units=n_classes, activation=None))
    # keras_model.add(Activation('softmax'))
    # keras_model.add(Dropout(0.1))
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
    parser.add_argument('-mc', '--max_cols_plot', dest='max_cols_plot',
                        default=MAX_COLS_PLOT, type=int,
                        help=f'Maxium cols to plot for feature importance (default: {MAX_COLS_PLOT}).')
    parser.add_argument('-bs', '--bootstrap_cols', dest='bootstrap_cols',
                        default=-1, type=int,
                        help=f'The number of cols to bootsrap from the dataframe (default: use all cols).')
    return parser.parse_args()


def run(args):
    # TODO: log out the args
    print(f'\n{args}')
    n_shuffles = args.n_shuffles
    corr_th = args.corr_th
    epoch = args.epoch
    batch = args.batch
    max_cols_plot = args.max_cols_plot

    # Create necessary dirs
    # dataset = DATAPATH_TR.split('_')[-1]  # TODO: clean/fix
    OUTDIR = os.path.join(file_path, f'results_{APP}_cor{corr_th}')
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    logger = set_logger(filename=os.path.join(OUTDIR, f'{APP}_main_logfile.log'))


    # ==========  Load data  ==========
    print('\n======= Load TC data =======')
    y_enc = pd.read_csv(YENC_PATH, sep='\t')
    data_train = pd.read_csv(DATAPATH_TR, sep='\t')
    data_val = pd.read_csv(DATAPATH_VL, sep='\t')
    print(f'\ndata_train.shape {data_train.shape}')
    print(f'data_val.shape   {data_val.shape}')
    
    if args.bootstrap_cols > -1:
        y_tmp = data_train.iloc[:,0]
        x_tmp = data_train.iloc[:,1:].sample(n=args.bootstrap_cols, axis=1, random_state=SEED)
        data_train = pd.concat([y_tmp, x_tmp], axis=1)
        data_val = data_val[data_train.columns]
    print(f'\ndata_train.shape {data_train.shape}')
    print(f'data_val.shape   {data_val.shape}')    

    # Compute corr matrix
    # cor = utils.compute_cor_mat(xvl, zero_diag=True, decimals=5)
    # fig = utils.plot_cor_heatmap(cor)
    # fig.savefig(os.path.join(OUTDIR, f'{APP}_feature_corr.png'), bbox_inches='tight')

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

    # ==========  RF classifier  ==========
    logger.info('------- Data for RF Classifier -------')
    xtr = data_train.iloc[:, 1:].copy()
    ytr = data_train.iloc[:, 0].copy().values
    xvl = data_val.iloc[:, 1:].copy()
    yvl = data_val.iloc[:, 0].copy().values
    features = xtr.columns
    print(f'\nnp.unique(ytr): {np.unique(ytr)}')
    logger.info(f'xtr.shape {xtr.shape}')
    logger.info(f'xvl.shape {xvl.shape}')
    logger.info(f'ytr.shape {ytr.shape}')
    logger.info(f'yvl.shape {yvl.shape}')

    # ---------- Train RF classifier ----------
    logger.info('------- Train RF Classifier -------')
    rf_model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features='sqrt', random_state=SEED)
    rf_model.fit(xtr, ytr)
    logger.info(f'Prediction score (mean accuracy): {rf_model.score(xvl, yvl):.4f}')

    yvl_preds = rf_model.predict(xvl)   
    #print('true', yvl[:10].values)
    print('true', yvl[:10])
    print('pred', yvl_preds[:10])
    logger.info('f1_score micro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='micro')))
    logger.info('f1_score macro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='macro')))

    utils.plot_confusion_matrix(y_true=yvl, y_pred=yvl_preds, labels=y_enc['label'].values,
                                title=f'{APP}_confusion_rf', savefig=True,
                                img_name=os.path.join(OUTDIR, f'{APP}_confusion_rf.png'))

    # ---------- MDI and PFI from RF ----------
    print('\n------- MDI and PFI from RF classifier -------')
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, max_cols_plot=max_cols_plot,
                                    title='RF Classifier (FI using MDI)', errorbars=False,
                                    plot_direction='v', color='darkorange')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_fi.png'), bbox_inches='tight')
    rf_fi = utils.get_rf_fi(rf_model, columns=features)
    rf_fi.to_csv(os.path.join(OUTDIR, f'{APP}_rf_fi.csv'), index=False)    

    # PFI
    t0 = time.time()
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles, y_enc=y_enc, outdir=OUTDIR)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c', verbose=True)
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(max_cols_plot=max_cols_plot, title='RF Classifier (PFI var)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_var.png'), bbox_inches='tight')

    fig = fi_obj.plot_score_fi(max_cols_plot=max_cols_plot, title='RF Classifier (PFI MDA: f1-score)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_score.png'), bbox_inches='tight')

    fig = fi_obj.plot_fimap(figsize=(20, 7), n_top_cols=10, title='RF PFI Map', drop_correlated=True)
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_map.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name=f'{APP}_rf')


    # ==========  NN classifier  ==========
    logger.info('                 ')
    logger.info('------- Data for NN Classifier -------')
    xtr = data_train.iloc[:, 1:].copy()
    ytr = data_train.iloc[:, 0].copy()
    xvl = data_val.iloc[:, 1:].copy()
    yvl = data_val.iloc[:, 0].copy()
    features = xtr.columns
    print(f'\nnp.unique(ytr): {np.unique(ytr)}')
    logger.info(f'xtr.shape {xtr.shape}')
    logger.info(f'xvl.shape {xvl.shape}')
    logger.info(f'ytr.shape {ytr.shape}')
    logger.info(f'yvl.shape {yvl.shape}')
    
    n_classes = len(np.unique(ytr))
    ytr = keras.utils.to_categorical(ytr, num_classes=n_classes)
    yvl = keras.utils.to_categorical(yvl, num_classes=n_classes)

     # ---------- Train NN classifier ----------
    logger.info('------- Train NN Classifier -------')
    keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    
    # callback_list = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, 
    #                                mode='auto', baseline=None, restore_best_weights=True)]
    callback_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
                     ModelCheckpoint(filepath=os.path.join(OUTDIR, f'{APP}_nn_model'),
                                     monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)]

    history = keras_model.fit(xtr, ytr, validation_data=(xvl, yvl),
                              epochs=epoch, batch_size=batch, verbose=1, callbacks=callback_list)
    # utils.plot_keras_learning(history, figsize = (10, 8), savefig=True,
    #                           img_name=os.path.join(OUTDIR, 'learning_with_lr'))

    score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    logger.info('Prediction score (val loss): {:.4f}'.format(score))
    
    yvl_preds = keras_model.predict(xvl)
    print('true', np.argmax(yvl[:10], axis=1))
    print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    logger.info('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    logger.info('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))
    
    # Reshape taregt (required for confusion matrix and PFI)
    if yvl_preds.ndim > 1 and yvl_preds.shape[1] > 1:  # if classification, get the class label
        yvl_preds = np.argmax(yvl_preds, axis=1)
    if yvl.ndim > 1 and yvl.shape[1] > 1:  # if classification, get the class label
        yvl = np.argmax(yvl, axis=1)

    utils.plot_confusion_matrix(y_true=yvl, y_pred=yvl_preds, labels=y_enc['label'].values,
                                title=f'{APP}_confusion_nn', savefig=True,
                                img_name=os.path.join(OUTDIR, f'{APP}_confusion_nn.png'))

    # PFI
    t0 = time.time()
    fi_obj = pfi.PFI(model=keras_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles, y_enc=y_enc, outdir=OUTDIR)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c', verbose=True)
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(max_cols_plot=max_cols_plot, title='NN Classifier (PFI var)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_nn_pfi_var.png'), bbox_inches='tight')

    fig = fi_obj.plot_score_fi(max_cols_plot=max_cols_plot, title='NN Classifier (PFI MDA: f1-score)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_nn_pfi_score.png'), bbox_inches='tight')

    fig = fi_obj.plot_fimap(figsize=(20, 7), n_top_cols=10, title='NN PFI Map', drop_correlated=True)
    fig.savefig(os.path.join(OUTDIR, f'{APP}_nn_pfi_map.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name=f'{APP}_nn')


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
