from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', 'utils_py'))
sys.path.append(utils_path)

from pfi import *
from pilot1_imports import *

DATAPATH_CLASSIFICATION = os.path.join('.', 'DATA', 'dataset_classification')
DATAPATH_REGRESSION = os.path.join('.', 'DATA', 'dataset_regression')
OUTDIR = os.path.join('.', 'results_test_keras')
N_SHUFFLES = 20
thres_xcorr = 0.9
SEED = 0


def create_nn_classifier(n_features, n_classes):
    keras_model = Sequential()
    keras_model.add(Dense(units=32, activation='relu', input_shape=(n_features,)))
    # keras_model.add(Dense(units=1, activation='sigmoid'))
    # keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    parser = argparse.ArgumentParser(description='Compute feature importance via feature shuffling (supports keras).')
    # parser.add_argument('-d', '--datapath', dest='datapath',
    #                     default=DATAPATH, type=str,
    #                     help='full path to the data')
    parser.add_argument('-ns', '--n_shuffles', dest='n_shuffles',
                        default=N_SHUFFLES, type=int,
                        help='number of shuffles (default: {})'.format(N_SHUFFLES))
    return parser.parse_args()


def run(args):
    print(args)
    # datapath = args.datapath
    n_shuffles = args.n_shuffles
    epochs = 80
    batch = 64

    # Create necessary dirs
    make_dir(OUTDIR)
    # os.makedirs(OUTDIR, exist_ok=True)  # python 3


    # ==========  RF classifier  ==========
    print('\nLoading classification dataset...')

    # Load data
    data = pd.read_csv(DATAPATH_CLASSIFICATION, sep='\t')
    xdata = data.iloc[:, 1:].values
    ydata = data.iloc[:, 0].values

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])
    print('np.unique(ydata)', np.unique(ydata))

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Drop cols (high correlation)
    xdata, corr, cols_dropped = drop_cols_on_xcorr(pd.DataFrame(xdata), thres_xcorr=thres_xcorr, verbose=True)
    print('xdata.shape', xdata.shape)
    print('xdata.columns', xdata.columns.tolist())

    # Split data
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)

    print('\nTraining RF Classifier...')
    rf_model = RandomForestClassifier(n_estimators=250, max_features='sqrt', min_samples_split=3, random_state=SEED)
    rf_model.fit(xtr, ytr)
    score = rf_model.score(xvl, yvl)
    print('Prediction score (mean accuracy): {:.4f}'.format(score))

    yvl_preds = rf_model.predict(xvl)
    print('true', yvl[:10])
    print('pred', yvl_preds[:10])
    print('f1_score micro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='micro')))
    print('f1_score macro: {:.3f}'.format(f1_score(y_true=yvl, y_pred=yvl_preds, average='macro')))

    # Plot FI
    indices, fig = plot_rf_fi(rf_model, n_features_toplot=None, title='FI RF Classifier')
    fi = get_fi_from_rf(rf_model)
    fi.to_csv(os.path.join(OUTDIR, 'rf_classifier_fi.csv'), index=False)
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_fi.png'), bbox_inches='tight')

    # PFI VAR
    fi_obj = PermutationFeatureImportance(model=rf_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=rf_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_var = fi_obj.compute_feature_importance_var()
    fi_var.to_csv(os.path.join(OUTDIR, 'rf_classifier_pfi_var.csv'), index=False)
    fig = fi_obj.plot_fi(title='RF Classifier (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_var.png'), bbox_inches='tight')

    # PFI SCORE
    fi_obj = PermutationFeatureImportance(model=rf_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=rf_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_score = fi_obj.compute_feature_importance_score(ml_type='c')
    fi_score.to_csv(os.path.join(OUTDIR, 'rf_classifier_pfi_score.csv'), index=False)
    fig = fi_obj.plot_fi(title='RF Classifier (PFI score: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, 'rf_classifier_pfi_score.png'), bbox_inches='tight')


    # ==========  RF regressor  ==========
    print('\nLoading regression dataset...')

    # Load data
    data = pd.read_csv(DATAPATH_REGRESSION, sep='\t')
    xdata = data.iloc[:, 1:].values
    ydata = data.iloc[:, 0].values

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Split data
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True)

    print('\nTraining RF Regressor...')
    rf_model = RandomForestRegressor(n_estimators=250, max_features='sqrt', min_samples_split=3, random_state=SEED)
    rf_model.fit(xtr, ytr)
    score = rf_model.score(xvl, yvl)
    print('Prediction score (r_square): {:.4f}'.format(score))

    # Plot FI
    indices, fig = plot_rf_fi(rf_model, n_features_toplot=None, title='FI RF Regressor')
    fi = get_fi_from_rf(rf_model)
    fi.to_csv(os.path.join(OUTDIR, 'rf_regressor_fi.csv'), index=False)
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_fi.png'), bbox_inches='tight')

    # PFI VAR
    fi_obj = PermutationFeatureImportance(model=rf_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=rf_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_var = fi_obj.compute_feature_importance_var()
    fi_var.to_csv(os.path.join(OUTDIR, 'rf_regressor_pfi_var.csv'), index=False)
    fig = fi_obj.plot_fi(title='RF Regressor (PFI var)', ylabel='Importance (relative)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_pfi_var.png'), bbox_inches='tight')

    # PFI SCORE
    fi_obj = PermutationFeatureImportance(model=rf_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=rf_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_score = fi_obj.compute_feature_importance_score(ml_type='r')
    fi_score.to_csv(os.path.join(OUTDIR, 'rf_regressor_pfi_score.csv'), index=False)
    fig = fi_obj.plot_fi(title='RF Regressor (PFI score: r^2)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, 'rf_regressor_pfi_score.png'), bbox_inches='tight')


    # ==========  NN classifier  ==========
    print('\nLoading classification dataset...')

    # Load data
    data = pd.read_csv(DATAPATH_CLASSIFICATION, sep='\t')
    xdata = data.iloc[:, 1:].values
    ydata = data.iloc[:, 0].values

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])
    print('np.unique(ydata)', np.unique(ydata))

    n_classes = len(np.unique(ydata))
    ydata = keras.utils.to_categorical(ydata, num_classes=n_classes)

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Drop cols (high correlation)
    xdata, corr, cols_dropped = drop_cols_on_xcorr(pd.DataFrame(xdata), thres_xcorr=thres_xcorr, verbose=True)
    print('xdata.shape', xdata.shape)
    print('xdata.columns', xdata.columns.tolist())

    # Split data
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)

    print('\nTraining NN Classifier...')
    keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    history = keras_model.fit(xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    print('Prediction score (val loss): {:.4f}'.format(score))

    yvl_preds = keras_model.predict(xvl)
    print('true', np.argmax(yvl[:10], axis=1))
    print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    print('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    print('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))

    # PFI VAR
    fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=keras_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_var = fi_obj.compute_feature_importance_var()
    fi_var.to_csv(os.path.join(OUTDIR, 'nn_classifier_pfi_var.csv'), index=False)
    fig = fi_obj.plot_fi(title='NN Classifier (PFI var)', ylabel='Importance (normalized)')
    fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_var.png'), bbox_inches='tight')

    # PFI SCORE
    fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=keras_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_score = fi_obj.compute_feature_importance_score(ml_type='c')
    fi_score.to_csv(os.path.join(OUTDIR, 'nn_classifier_pfi_score.csv'), index=False)
    fig = fi_obj.plot_fi(title='NN Classifier (PFI score: f1-score)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, 'nn_classifier_pfi_score.png'), bbox_inches='tight')


    # ==========  NN regressor  ==========
    print('\nLoading regression dataset...')

    # Load data
    data = pd.read_csv(DATAPATH_REGRESSION, sep='\t')
    xdata = data.iloc[:, 1:].values
    ydata = data.iloc[:, 0].values

    print('data.shape', data.shape)
    print(data.iloc[:3, :4])

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Split data
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True)

    print('\nTraining NN Regressor...')
    keras_model = create_nn_regressor(n_features=xtr.shape[1])
    history = keras_model.fit(xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    print('Prediction score (val loss): {:.4f}'.format(score))

    # PFI VAR
    fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=keras_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_var = fi_obj.compute_feature_importance_var()
    fi_var.to_csv(os.path.join(OUTDIR, 'nn_regressor_pfi_var.csv'), index=False)
    fig = fi_obj.plot_fi(title='NN Regressor (PFI var)', ylabel='Importance (normalized)')
    fig.savefig(os.path.join(OUTDIR, 'nn_regressor_pfi_var.png'), bbox_inches='tight')

    # PFI SCORE
    fi_obj = PermutationFeatureImportance(model=keras_model, xdata=pd.DataFrame(xtr), ydata=ytr, n_shuffles=n_shuffles)
    # fi_obj = FeatureImportanceShuffle(model=keras_model, xdata=pd.DataFrame(xvl), ydata=yvl, n_shuffles=n_shuffles)
    fi_scofe = fi_obj.compute_feature_importance_score(ml_type='r')
    fi_score.to_csv(os.path.join(OUTDIR, 'nn_regressor_pfi_score.csv'), index=False)
    fig = fi_obj.plot_fi(title='NN Regressor (PFI score: r^2)', ylabel='Importance (score decrease)')
    fig.savefig(os.path.join(OUTDIR, 'nn_regressor_pfi_score.png'), bbox_inches='tight')


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
