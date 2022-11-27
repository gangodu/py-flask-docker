#!/usr/bin/python3
"""
Train Python base Predictions
"""
import os
import pandas as pd

from joblib import dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

def train():
    """ Load directory paths for persisting model """

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
    y_train = data_train['# Letter'].values
    x_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)
    print("Shape of the training data")
    print(x_train.shape)
    print(y_train.shape)

    # Data normalization (0,1)
    x_train = preprocessing.normalize(x_train, norm='l2')

    # Models training

    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(x_train, y_train)

    # Serialize model
    dump(clf_lda, MODEL_PATH_LDA)

    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(x_train, y_train)

    # Serialize model
    dump(clf_NN, MODEL_PATH_NN)

if __name__ == '__main__':
    train()
