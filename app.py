import pandas as pd
from flask import Flask, jsonify, request
import pickle
from keras.models import model_from_json,model_from_yaml
import clean_data as cd
import json
import requests
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("model.h5")
# app
app = Flask(__name__)
# routes
@app.route('/', methods=['GET'])
def home():
    return ("Hello world!")
@app.route('/predict', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    #print('Data received: "{data}"'.format(data=data))
    #data=json.dumps(d)
    data = data["text"]
    # predictions
    result = cd.get_sentiment(model, data)
    #result = json.dumps(result)

    # send back to browser
    output = result
    temp = sorted(result.items(), key=lambda item: item[1], reverse=True)
    result = dict(temp[0:3])

    # return data
    #return data
    return jsonify(result)
if __name__ == '__main__':
   app.run(debug=true)
