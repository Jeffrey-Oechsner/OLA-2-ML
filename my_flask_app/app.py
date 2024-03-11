from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVR, SVC

app = Flask(__name__)

# Load the trained model (Make sure 'model.joblib' is in the same directory as this file or provide the full path)
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = 'No prediction made yet.'  # Default message
    try:
        # Extracting each feature from the form, excluding 'year'
        crimes_penal_code = request.form.get('crimes_penal_code', type=int)
        crimes_person = request.form.get('crimes_person', type=int)
        murder = request.form.get('murder', type=int)
        assault = request.form.get('assault', type=int)
        sexual_offenses = request.form.get('sexual_offenses', type=int)
        rape = request.form.get('rape', type=int)
        stealing_general = request.form.get('stealing_general', type=int)
        burglary = request.form.get('burglary', type=int)
        house_theft = request.form.get('house_theft', type=float)
        vehicle_theft = request.form.get('vehicle_theft', type=float)
        out_of_vehicle_theft = request.form.get('out_of_vehicle_theft', type=float)
        shop_theft = request.form.get('shop_theft', type=float)
        robbery = request.form.get('robbery', type=int)
        fraud = request.form.get('fraud', type=int)
        criminal_damage = request.form.get('criminal_damage', type=int)
        other_penal_crimes = request.form.get('other_penal_crimes', type=int)
        narcotics = request.form.get('narcotics', type=float)
        drunk_driving = request.form.get('drunk_driving', type=int)
        population = request.form.get('population', type=int)

        # Combining all features into an array for prediction, excluding 'year'
        features = np.array([[crimes_penal_code, crimes_person, murder, assault, sexual_offenses, rape, stealing_general,
                              burglary, house_theft, vehicle_theft, out_of_vehicle_theft, shop_theft, robbery, fraud,
                              criminal_damage, other_penal_crimes, narcotics, drunk_driving, population]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Format the prediction for displaying
        prediction_text = f'The predicted total crimes: {prediction[0]}'
    except Exception as e:
        prediction_text = f'Error making prediction: {e}'

    # Render the results page with the prediction result
    return render_template('results.html', prediction=prediction_text)



# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extracting each feature from the form, excluding 'year'
#         crimes_penal_code = request.form.get('crimes_penal_code', type=int)
#         crimes_person = request.form.get('crimes_person', type=int)
#         murder = request.form.get('murder', type=int)
#         assault = request.form.get('assault', type=int)
#         sexual_offenses = request.form.get('sexual_offenses', type=int)
#         rape = request.form.get('rape', type=int)
#         stealing_general = request.form.get('stealing_general', type=int)
#         burglary = request.form.get('burglary', type=int)
#         house_theft = request.form.get('house_theft', type=float)
#         vehicle_theft = request.form.get('vehicle_theft', type=float)
#         out_of_vehicle_theft = request.form.get('out_of_vehicle_theft', type=float)
#         shop_theft = request.form.get('shop_theft', type=float)
#         robbery = request.form.get('robbery', type=int)
#         fraud = request.form.get('fraud', type=int)
#         criminal_damage = request.form.get('criminal_damage', type=int)
#         other_penal_crimes = request.form.get('other_penal_crimes', type=int)
#         narcotics = request.form.get('narcotics', type=float)
#         drunk_driving = request.form.get('drunk_driving', type=int)
#         population = request.form.get('population', type=int)

#         # Combining all features into an array for prediction, excluding 'year'
#         features = np.array([[crimes_penal_code, crimes_person, murder, assault, sexual_offenses, rape, stealing_general,
#                               burglary, house_theft, vehicle_theft, out_of_vehicle_theft, shop_theft, robbery, fraud,
#                               criminal_damage, other_penal_crimes, narcotics, drunk_driving, population]])
        
#         # Make prediction
#         prediction = model.predict(features)
        
#         # Format the prediction for displaying
#         prediction_text = f'The predicted total crimes: {prediction[0]}'
#     except Exception as e:
#         prediction_text = f'Error making prediction: {e}'

#     # Render the results page with the prediction result
#     return render_template('results.html', prediction=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
