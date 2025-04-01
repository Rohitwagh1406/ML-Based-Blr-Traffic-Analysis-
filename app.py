
from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained models
traffic_volume_model = RandomForestRegressor(random_state=42)
congestion_level_model = RandomForestClassifier(random_state=42)
scaler = StandardScaler()

@app.route('/predict_traffic_volume', methods=['POST'])
def predict_traffic_volume():
    data = request.get_json()
    features = np.array([[
        data['average_speed'],
        data['travel_time_index'],
        data['road_capacity_utilization'],
        data['environmental_impact'],
        data['public_transport_usage'],
        data['traffic_signal_compliance'],
        data['parking_usage'],
        data['pedestrian_and_cyclist_count'],
        data['weather_conditions'],
        data['roadwork_and_construction_activity'],
        data['year'],
        data['month'],
        data['day'],
        data['weekday']
    ]])
    
    features_scaled = scaler.transform(features)
    traffic_volume_prediction = traffic_volume_model.predict(features_scaled)
    
    return jsonify({'predicted_traffic_volume': traffic_volume_prediction[0]})

@app.route('/predict_congestion_level', methods=['POST'])
def predict_congestion_level():
    data = request.get_json()
    features = np.array([[
        data['average_speed'],
        data['travel_time_index'],
        data['road_capacity_utilization'],
        data['environmental_impact'],
        data['public_transport_usage'],
        data['traffic_signal_compliance'],
        data['parking_usage'],
        data['pedestrian_and_cyclist_count'],
        data['weather_conditions'],
        data['roadwork_and_construction_activity'],
        data['year'],
        data['month'],
        data['day'],
        data['weekday']
    ]])
    
    features_scaled = scaler.transform(features)
    congestion_level_prediction = congestion_level_model.predict(features_scaled)
    
    return jsonify({'predicted_congestion_level': str(congestion_level_prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
