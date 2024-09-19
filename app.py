# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model1 = joblib.load('Delay_model.pkl')
model2 = joblib.load('Cost_Overrun_model.pkl')
model3 = joblib.load('Injury_model.pkl')

@app.route('/delay_predict', methods=['GET' , 'POST'])
def prediction1():
    data1 = request.json  # Expecting JSON request
    features1 = np.array(data1['features'])  # Expects a list of features in the body
    prediction1 = model1.predict([features1])
    return jsonify({'prediction': int(prediction1[0])})

@app.route('/cost_predict', methods=['GET' , 'POST'])
def prediction2():
    data2 = request.json  # Expecting JSON request
    features2 = np.array(data2['features'])  # Expects a list of features in the body
    prediction2 = model2.predict([features2])
    return jsonify({'prediction': int(prediction2[0])})

@app.route('/injury_predict', methods=['GET' , 'POST'])
def prediction3():
    data3 = request.json  # Expecting JSON request
    features3 = np.array(data3['features'])  # Expects a list of features in the body
    prediction3 = model3.predict([features3])
    return jsonify({'prediction': int(prediction3[0])})

if __name__ == '__main__':
    app.run(debug=True)
