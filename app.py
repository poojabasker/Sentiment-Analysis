import pandas as pd
from flask import Flask, jsonify, request
import pickle
from keras.models import model_from_json, model_from_yaml
import clean_data as cd
import json
import requests
import retrain as rt
#from flask_ngrok import run_with_ngrok

yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("model.h5")

# app
app = Flask(__name__)
# run_with_ngrok(app)

# routes
@app.route('/', methods=['GET'])
def get():
    return("hello world")

@app.route('/retrain', methods=['GET'])
def retrain():
    rt.get_sentiment()
    return

@app.route('/predict', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    #print('Data received: "{data}"'.format(data=data))
    # data=json.dumps(d)
    data = data["text"]

    # predictions
    result = cd.get_sentiment(model, data)

    temp = sorted(result.items(), key=lambda item: item[1], reverse=True)
    final = dict(temp[0:3])

    #return data
    return jsonify(final)

@app.route('/predictall', methods=['POST'])
def predictall():
    # get data
    data = request.get_json(force=True)
    #print('Data received: "{data}"'.format(data=data))
    # data=json.dumps(d)
    data = data["text"]

    # predictions
    result = cd.get_sentiment(model, data)

    #return data
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
