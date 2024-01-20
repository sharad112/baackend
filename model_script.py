import os
from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier


current_dir=os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

# Load the model
model_path=os.path.join(current_dir,'ai_model',"test.joblib")
model=joblib.load(model_path)
print("hello")
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming your model expects input features in a specific format
    features = data['features']
    prediction = model.predict(features)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000)

 