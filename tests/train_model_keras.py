"""
This code trains a keras NN model and saves the model and its weights to files.
The model and weights can then loaded for inference (used in test_pfi_master.py).
"""
from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import keras
from keras.models import Sequential
from keras.layers import Dense

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils


# DATAPATH_CLASSIFICATION = os.path.join(file_path, 'data', 'data_classification')
# DATAPATH_REGRESSION = os.path.join(file_path, 'data', 'data_regression')
DATAPATH_CLASSIFICATION = os.path.join(file_path, 'data', 'data_classification_corr')
DATAPATH_REGRESSION = os.path.join(file_path, 'data', 'data_regression_corr')
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
    keras_model.add(Dense(units=32, activation='relu', input_shape=(n_features,)))
    keras_model.add(Dense(units=1, activation=None))
    keras_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return keras_model


def init_params():
    parser = argparse.ArgumentParser(description='Create, train, and store keras model.')
    parser.add_argument('-e', '--epochs', dest='epochs',
                        default=100, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch', dest='batch',
                        default=64, type=int,
                        help='batch size')
    return parser.parse_args()


def run(args):
    print(args)
    epochs = args.epochs
    batch = args.batch


    # ==========  NN classifier  ==========
    print('\nLoading classification dataset...')

    # Load data
    data = pd.read_csv(DATAPATH_CLASSIFICATION, sep='\t')
    print('data.shape', data.shape)
    print(data.iloc[:3, :5])

    xdata = data.iloc[:, 1:].values
    ydata = data.iloc[:, 0].values
    print('np.unique(ydata)', np.unique(ydata))

    n_classes = len(np.unique(ydata))
    ydata = keras.utils.to_categorical(ydata, num_classes=n_classes)

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Split data
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)

    print('\nTrain NN Classifier ...')
    keras_model = create_nn_classifier(n_features=xtr.shape[1], n_classes=n_classes)
    history = keras_model.fit(xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    print('Prediction score (val loss): {:.4f}'.format(score))

    yvl_preds = keras_model.predict(xvl)
    print('true', np.argmax(yvl[:10], axis=1))
    print('pred', np.argmax(yvl_preds[:10, :], axis=1))
    print('f1_score micro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='micro')))
    print('f1_score macro: {:.3f}'.format(f1_score(y_true=np.argmax(yvl, axis=1), y_pred=np.argmax(yvl_preds, axis=1), average='macro')))

    # Create dir to save the model
    MODELDIR = os.path.join(file_path, 'keras_model_classifier')
    utils.make_dir(MODELDIR)

    # Save initial model
    print('\nSave keras model (classifier) ...')
    model_json = keras_model.to_json()
    model_path = os.path.join(MODELDIR, 'keras_model.json')
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)

    # Save the initialized weights to HDF5
    print('Saving keras weights...')
    weights_path = os.path.join(MODELDIR, 'keras_weights.h5')
    keras_model.save_weights(weights_path)


    # ==========  NN regressor  ==========
    print('\nLoad regression dataset ...')

    # Load data
    data = pd.read_csv(DATAPATH_REGRESSION, sep='\t')
    print('data.shape', data.shape)
    print(data.iloc[:3, :5])

    ydata = data.iloc[:, 0].values
    xdata = data.iloc[:, 1:].values

    # Scale data
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Split data
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True)

    print('\nTrain NN Regressor ...')
    keras_model = create_nn_regressor(n_features=xtr.shape[1])
    history = keras_model.fit(xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    score = keras_model.evaluate(xvl, yvl, verbose=False)[-1]  # compute the val loss
    print('Prediction score (val loss): {:.4f}'.format(score))

    # Create dir to save the model
    MODELDIR = os.path.join(file_path, 'keras_model_regressor')
    utils.make_dir(MODELDIR)

    # Save initial model
    print('\nSave keras model (regressor) ...')
    model_json = keras_model.to_json()
    model_path = os.path.join(MODELDIR, 'keras_model.json')
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)

    # Save the initialized weights to HDF5
    print('Save keras weights ...')
    weights_path = os.path.join(MODELDIR, 'keras_weights.h5')
    keras_model.save_weights(weights_path)


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()
