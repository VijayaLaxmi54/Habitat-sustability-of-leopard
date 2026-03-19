from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models and scaler
MODEL_PATH = 'models/rf_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None
    print("Warning: Model or scaler not found. Please run train.py first.")

@app.route('/')
def home():
    has_model = model is not None
    return render_template('index.html', has_model=has_model)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not trained yet!'}), 500
        
    try:
        # Get data from form
        ndvi = float(request.form.get('ndvi', 0))
        elevation = float(request.form.get('elevation', 0))
        slope = float(request.form.get('slope', 0))
        land_use = int(request.form.get('land_use', 1))
        dist_water = float(request.form.get('dist_water', 0))
        dist_roads = float(request.form.get('dist_roads', 0))
        dist_settlements = float(request.form.get('dist_settlements', 0))
        
        # Prepare feature array exactly in the order they were trained
        # ['NDVI', 'Elevation', 'Slope', 'Land_Use', 'Distance_to_Water', 'Distance_to_Roads', 'Distance_to_Settlements']
        features = np.array([[ndvi, elevation, slope, land_use, dist_water, dist_roads, dist_settlements]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] # Probability of being suitable
        
        result = {
            'class': int(prediction),
            'suitability': 'Suitable' if prediction == 1 else 'Unsuitable',
            'probability': float(probability),
            'probability_pct': f"{probability * 100:.1f}%"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
