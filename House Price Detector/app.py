from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Manually load the mean and standard deviation for normalization
scaler = StandardScaler()
scaler.mean_ = np.load('mean.npy')
scaler.scale_ = np.load('std.npy')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from user input in JSON format
    data = request.get_json()

    # Extract user input
    house_age = data['house_age']
    distance_to_mrt = data['distance_to_mrt']
    number_of_stores = data['number_of_stores']
    latitude = data['latitude']
    longitude = data['longitude']
    area = data['area']  # The user-provided area for the house

    # Prepare the input feature vector
    input_features = np.array([[house_age, distance_to_mrt, number_of_stores, latitude, longitude,area]])

    # Normalize the input features using the manually loaded scaler
    input_scaled = scaler.transform(input_features)

    # Predict the price per unit area
    predicted_unit_price = model.predict(input_scaled)[0]

    # Calculate the total price by multiplying the predicted unit price by the area
    total_price = predicted_unit_price * area

    # Return the result as a JSON response
    return jsonify({
        'predicted_unit_price': predicted_unit_price,
        'total_price': total_price
    })

if __name__ == '__main__':
    app.run(debug=True)
