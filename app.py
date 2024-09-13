# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('Delay_model.pkl')

@app.route('/predict', methods=['GET' , 'POST'])
def predict():
    data = request.json  # Expecting JSON request
    features = np.array(data['features'])  # Expects a list of features in the body
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
