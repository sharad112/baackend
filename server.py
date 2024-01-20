from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from predict import crop_prediction_dt, crop_prediction_rf, RandomForestClassifier, DecisionTree
app = Flask(__name__)
CORS(app, resources={r"/.*": {"origins": "*"}})

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
    
        # Get the input data from the request
        input_data = request.get_json()
        input_data=input_data["input_feature"]
        print(input_data)
        # Extract input features from the JSON data
        N = int(input_data['N'])
        print("line 24")
        P = int(input_data['P'])
        K = int(input_data['K'])
        temperature = int(input_data['temperature'])
        humidity = int(input_data['humidity'])
        ph = int(input_data['ph'])
        rainfall = int(input_data['rainfall'])
        print("line 30")
        # Make the prediction using the loaded model
        new_data_point = np.array([N, P, K, temperature, humidity, ph, rainfall])
         # Predict crop using Random Forest
        predicted_crop_rf = crop_prediction_rf(new_data_point)

        # Predict crop using Decision Tree
        predicted_crop_dt = crop_prediction_dt(new_data_point)

        
        # Return the prediction as JSON response
        response = {'predicted_crop': predicted_crop_rf}
        print(response)
        return jsonify(response,"RES")

    except Exception as e:
        # Handle errors gracefully and return an error response
        print(e)
        error_message = f"Error: {str(e)}"
        return jsonify({'error': error_message})


if __name__ == '__main__':
    # Run the Flask app on port 5000
    print(crop_prediction_rf(np.array([12, 12, 12, 12, 12, 12, 12])))
    print(crop_prediction_dt(np.array([12, 12, 12, 12, 12, 12, 12])))
    app.run(port=5000, debug=True)
