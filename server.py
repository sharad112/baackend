from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
app = Flask(__name__)
CORS(app, resources={r"/.*": {"origins": "*"}})

from sklearn.ensemble import RandomForestClassifier
# rf_classifier_loaded = RandomForestClassifier('ai_model/test.jblib')  # Load your model here
loaded_model = joblib.load('random_forest_model.joblib')
loaded_label_mapping = joblib.load('label_mapping.joblib')

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
        N = input_data['N']
        print("line 24")
        P = input_data['P']
        K = input_data['K']
        temperature = input_data['temperature']
        humidity = input_data['humidity']
        ph = input_data['ph']
        rainfall = input_data['rainfall']
        print("line 30")
        # Make the prediction using the loaded model
        new_data_point = np.array([N, P, K, temperature, humidity, ph, rainfall])
        crop_label = loaded_model.predict(new_data_point)[0]

    # Map the label to the crop name using the label_mapping
        predicted_crop = loaded_label_mapping.get(crop_label)

        
        # Return the prediction as JSON response
        response = {'predicted_crop': predicted_crop}
        print(response)
        return jsonify(response,"RES")

    except Exception as e:
        # Handle errors gracefully and return an error response
        print(e)
        error_message = f"Error: {str(e)}"
        return jsonify({'error': error_message})


if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(port=5000, debug=True)
