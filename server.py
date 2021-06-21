
import io
import json
import pandas as pd
import ast
from preprocess import prepro
from preprocess_new import prepro_new

from flask import Flask, jsonify, request, render_template
from pycaret.classification import *

app = Flask(__name__)
model_full = load_model('modell_full_saved_20210610')
test_model_full = load_model('model_full_saved_20210618')
set_id = 0


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
    prediction = predict_model(model_full, data=prepro(model_in), raw_score=True)
    return jsonify({'data': list(prediction['Label'])})

@app.route('/test', methods=['GET'])
def test():
    return render_template('test_index.html')

@app.route('/test_get_score', methods=['POST'])
def test_get_score():
    data = request.json
    print(data)
    data = data[0]
    answered = [i for i, v in enumerate(data[9:]) if v != '-1']
    model_in = []
    for i in data:
        if(i == '-1'):
            model_in.append(None)
        else:
            model_in.append(i)    
    prediction = predict_model(test_model_full, data=prepro_new(model_in), raw_score=True)
    recommended_index = prediction.loc[~(prediction.index.isin(answered)),['Score_0', 'Score_1', 'Score_2', 'Score_3']].std(axis = 1).idxmin()
    model_pred = prediction.loc[prediction.index == recommended_index,'Label'].values[0]
    return jsonify({'data': [str(recommended_index), str(model_pred)]})

@app.route('/get_userid', methods=['POST'])
def get_userid():
    data = request.json
    print(data)
    global set_id
    set_id += 1
    if(set_id >= 1000000):
        set_id = 1
    
    return jsonify({'data': [str(set_id)]})

@app.route('/last_query', methods=['POST'])
def last_query():
    data = request.json
    print(data)
    return jsonify({'data': ['-1']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
