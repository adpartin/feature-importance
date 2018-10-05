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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils

APP = 'tc'
# DATAPATH = os.path.join(file_path, 'data', f'{APP}_data')
# DATAPATH_TR = os.path.join(file_path, 'data', f'{APP}_data_train')
# DATAPATH_VL = os.path.join(file_path, 'data', f'{APP}_data_val')
DATAPATH_TR = os.path.join('/vol/ml/apartin/Benchmarks/Data/Pilot1', 'P1B1.dev.train.csv')
DATAPATH_VL = os.path.join('/vol/ml/apartin/Benchmarks/Data/Pilot1', 'P1B1.dev.test.csv')
# YENC_PATH = os.path.join(file_path, 'data', f'{APP}_y_enc')
N_SHUFFLES = 20
CORR_THRES = 0.9
EPOCH = 60
BATCH = 32
MAX_COLS = 20
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
    keras_model = Sequential()
    keras_model.add(Dense(units=32, activation='relu', input_shape=(n_features,)))
    keras_model.add(Dense(units=32, activation='relu'))
    # if n_classes == 2:
    #     keras_model.add(Dense(units=1, activation='sigmoid'))
    #     keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # elif n_classes > 2:
    
    # keras_model.add(Activation('softmax'))
    # keras_model.add(Dropout(0.2))
    # keras_model.add(Dense(units=n_classes, activation=None))
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
                        help=f'The number of cols to bootsrap from the dataframe (default: use all cols).')
    return parser.parse_args()


def run(args):
    # TODO: log out the args
    print(args)
    n_shuffles = args.n_shuffles
    corr_th = args.corr_th
    epoch = args.epoch
    batch = args.batch
    max_cols = args.max_cols

    # Create necessary dirs
    OUTDIR = os.path.join(file_path, f'results_aacr_{APP}_ff_cor{corr_th}')
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    logger = set_logger(filename=os.path.join(OUTDIR, f'{APP}_main_logfile.log'))


    # ==========  Load data  ==========
    print('\nLoad TC data ...')

    # ---------- Load data ----------
    # y_enc = pd.read_csv(YENC_PATH, sep='\t')
    ## data = pd.read_csv(DATAPATH, sep='\t')
    ## xdata = data.iloc[:, 1:].copy()
    ## ydata = data.iloc[:, 0].copy()
    data_train = pd.read_csv(DATAPATH_TR, sep=',')
    data_val = pd.read_csv(DATAPATH_VL, sep=',')
    print(f'\ndata_train.shape {data_train.shape}')
    print(f'data_val.shape   {data_val.shape}')
    
    mm = pd.read_csv('/vol/ml/apartin/Benchmarks/Data/Pilot1/lincs1000.tsv', sep='\t')

    train = train[['case_id', 'cancer_type'] + mm['gdc'].tolist()]  # Extract lincs from the whole dataset
    test = test[['case_id', 'cancer_type'] + mm['gdc'].tolist()]  # Extract lincs from the whole dataset
    print(train.shape)
    print(test.shape)

    if args.bootstrap_cols > -1:
        ## xdata = xdata.sample(n=args.bootstrap_cols, axis=1, random_state=SEED)  # Take a subset of cols
        y_tmp = data_train.iloc[:,0]
        x_tmp = data_train.iloc[:,1:].sample(n=args.bootstrap_cols, axis=1, random_state=SEED)
        data_train = pd.concat([y_tmp, x_tmp], axis=1)
        data_val = data_val[data_train.columns]
    print(f'\ndata_train.shape {data_train.shape}')
    print(f'data_val.shape   {data_val.shape}')    
    ##features = xdata.columns

    ##print('data.shape', data.shape)
    ##print(data.iloc[:3, :4])

    ##print('\nxdata.shape', xdata.shape)
    ##print('np.unique(ydata)', np.unique(ydata))

    ##scaler = StandardScaler()
    ##xdata = scaler.fit_transform(xdata)
    ##xdata = pd.DataFrame(xdata, columns=features)

    ##xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)

    # ==========  RF classifier  ==========
    logger.info('RF classifier ...')
    logger.info('-----------------')

    # ---------- Get the data ----------
    xtr = data_train.iloc[:, 1:].copy()
    ytr = data_train.iloc[:, 0].copy()
    xvl = data_val.iloc[:, 1:].copy()
    yvl = data_val.iloc[:, 0].copy()
    features = xtr.columns
    logger.info(f'xtr.shape {xtr.shape}')
    logger.info(f'xvl.shape {xvl.shape}')
    logger.info(f'ytr.shape {ytr.shape}')
    logger.info(f'yvl.shape {yvl.shape}')

    # ---------- Train RF classifier ----------
    logger.info(f'Train RF Classifier ...')
    rf_model = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=SEED)
    rf_model.fit(xtr, ytr)
    logger.info(f'Prediction score (mean accuracy): {rf_model.score(xvl, yvl):.4f}')

    yvl_preds = rf_model.predict(xvl)   
    print('true', yvl[:10].values)
    print('pred', yvl_preds[:10])
    logger.info('f1_score micro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='micro')))
    logger.info('f1_score macro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='macro')))

    # TODO: finish this ...
    # df_conf = utils.plot_confusion_matrix(y_true=yvl, y_pred=yvl_preds, labels=y_enc['type'].values,
    #                                       title=f'{APP}_confusion', savefig=True, img_name=f'{APP}_confusion')

    # ---------- Feature importance ----------
    # Plot RF FI
    indices, fig = utils.plot_rf_fi(rf_model, columns=features, max_cols=max_cols, title='RF Classifier (FI using MDI)')
    rf_fi = utils.get_rf_fi(rf_model, columns=features)
    rf_fi.to_csv(os.path.join(OUTDIR, f'{APP}_rf_fi.csv'), index=False)
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_fi.png'), bbox_inches='tight')

    # PFI
    logger.info('Compute PFI ...')
    t0 = time.time()
    fi_obj = pfi.PFI(model=rf_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles, outdir=OUTDIR)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c', verbose=True)
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(max_cols=max_cols, title='RF Classifier (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_var.png'), bbox_inches='tight')
    fig = fi_obj.plot_score_fi(max_cols=max_cols, title='RF Classifier (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_rf_pfi_score.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name=f'{APP}_rf')


    # ==========  NN classifier  ==========
    logger.info('                 ')
    logger.info('NN classifier ...')
    logger.info('-----------------')

    # ---------- Get the data ----------
    xtr = data_train.iloc[:, 1:].copy()
    ytr = data_train.iloc[:, 0].copy()
    xvl = data_val.iloc[:, 1:].copy()
    yvl = data_val.iloc[:, 0].copy()
    features = xtr.columns
    logger.info(f'xtr.shape {xtr.shape}')
    logger.info(f'xvl.shape {xvl.shape}')
    logger.info(f'ytr.shape {ytr.shape}')
    logger.info(f'yvl.shape {yvl.shape}')
    
    n_classes = len(np.unique(ytr))
    ytr = keras.utils.to_categorical(ytr, num_classes=n_classes)
    yvl = keras.utils.to_categorical(yvl, num_classes=n_classes)

     # ---------- Train NN classifier ----------
    logger.info('Training NN Classifier...')
    keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    history = keras_model.fit(xtr, ytr, epochs=epoch, batch_size=batch, verbose=0)
    # utils.plot_keras_learning(history, figsize = (10, 8), savefig=True,
    #                           img_name=os.path.join(OUTDIR, 'learning_with_lr'))
    score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    logger.info('Prediction score (val loss): {:.4f}'.format(score))
    
    yvl_preds = keras_model.predict(xvl)
    print('true', np.argmax(yvl[:10], axis=1))
    print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    logger.info('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    logger.info('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))
    
    # ---------- Feature importance ----------
    # PFI
    logger.info('Compute PFI ...')
    t0 = time.time()
    fi_obj = pfi.PFI(model=keras_model, xdata=xvl, ydata=yvl, n_shuffles=n_shuffles, outdir=OUTDIR)
    fi_obj.gen_col_sets(th=corr_th, toplot=False)
    fi_obj.compute_pfi(ml_type='c', verbose=True)
    print(f'Total PFI time:  {(time.time()-t0)/60:.3f} mins')

    # Plot and save PFI
    fig = fi_obj.plot_var_fi(max_cols=max_cols, title='NN Classifier (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_nn_pfi_var.png'), bbox_inches='tight')
    fig = fi_obj.plot_score_fi(max_cols=max_cols, title='NN Classifier (PFI MDA: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, f'{APP}_nn_pfi_score.png'), bbox_inches='tight')

    # Dump resutls
    fi_obj.dump(path=OUTDIR, name=f'{APP}_nn')


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
