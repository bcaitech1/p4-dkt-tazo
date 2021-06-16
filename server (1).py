
import io
import json
import pandas as pd
import ast
from preprocess import prepro

from flask import Flask, jsonify, request, render_template
from pycaret.classification import *

app = Flask(__name__)
model_full = load_model('modell_full_saved_20210610')


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = ast.literal_eval(request.data.decode('utf-8'))
        prediction = predict_model(model_full, data=prepro(data), raw_score=True)
        return jsonify({'data': list(prediction['Label'])})

@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    model_in = ['m']
    for i in range(9):
        model_in.append(data[i]['answer'])
    #model_in = ['m', '2', '1', '2', '2', '1', '0', '0', '0', '0']
    prediction = prediction = predict_model(model_full, data=prepro(model_in), raw_score=True)
    return jsonify({'data': list(prediction['Label'])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
