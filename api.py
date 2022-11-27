#!/usr/bin/python3
"""
Python program to access trained model in real time using APIs
"""
import json
import os
import pandas as pd

from joblib import load
from flask import Flask

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

# Loading LDA model
print("Loading model from: {}", MODEL_PATH_LDA)
inference_lda = load(MODEL_PATH_LDA)

# loading Neural Network model
print("Loading model from: {}", MODEL_PATH_NN)
inference_NN = load(MODEL_PATH_NN)

app = Flask(__name__)


@app.route('/line/<lines>')
def line(lines):
    """
    Get data from json and return the requested row defined by the variable Line
    """
    with open('./test.json', 'r', 100, "UTF-8") as jsonfile:
        file_data = json.loads(jsonfile.read())
    return json.dumps(file_data[lines])

@app.route('/prediction/<int:lines>',methods=['POST', 'GET'])
def prediction(lines):
    """
    Return prediction for both NN and LDA inference model with the requested row as input
    """
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    x_data = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    x_test = x_data.iloc[lines,:].values.reshape(1, -1)
    clf_lda = load(MODEL_PATH_LDA)
    prediction_lda = clf_lda.predict(x_test)
    clf_nn = load(MODEL_PATH_NN)
    prediction_nn = clf_nn.predict(x_test)
    return {'prediction LDA': int(prediction_lda), 'prediction Neural Network': int(prediction_nn)}

@app.route('/score',methods=['POST', 'GET'])
def score():
    """     Return classification score for both NN and LDA inference model   """
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    y_test = data_test['# Letter'].values
    x_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    clf_lda = load(MODEL_PATH_LDA)
    score_lda = clf_lda.score(x_test, y_test)
    clf_nn = load(MODEL_PATH_NN)
    score_nn = clf_nn.score(x_test, y_test)
    return {'Score LDA': score_lda, 'Score Neural Network': score_nn}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    