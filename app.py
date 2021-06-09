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
model.load_weights("model.h5")

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['GET'])
def home():
    return ("Hello world!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data = data["text"]
    result = cd.get_sentiment(model, data)
    
    temp = sorted(result.items(), key=lambda item: item[1], reverse=True)
    result = dict(temp[0:3])

    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=true)
