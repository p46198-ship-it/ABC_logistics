import joblib
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('logi.sav')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert dictionary to DataFrame. Ensure column order matches training data.
        # The order of columns was: 'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
        # 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
        # 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
        # 'Warehouse_Processing_Time'
        feature_names = [
            'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
            'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
            'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
            'Warehouse_Processing_Time'
        ]
        input_df = pd.DataFrame([data], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_proba_class_0': float(prediction_proba[0][0]),
            'prediction_proba_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local development, use app.run(debug=True)
    # To run in Colab or a similar environment and expose via ngrok:
    # 1. Install Flask: !pip install Flask
    # 2. Run this cell.
    # 3. In a new cell, run:
    #    !pip install pyngrok
    #    from pyngrok import ngrok
    #    # Terminate any previous ngrok tunnels
    #    ngrok.kill()
    #    # Open a HTTP tunnel on port 5000
    #    public_url = ngrok.connect(5000)
    #    print(f" * ngrok tunnel available at: {public_url}")
    #    # Now you can make requests to public_url/predict
    
    # For deployment, consider using a production-ready WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000)
