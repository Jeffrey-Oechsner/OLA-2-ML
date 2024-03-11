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
# Extract input features from the posted form
    # Here, I'm assuming 'year' is one of the input fields in your form
    # You need to replace 'year' and its handling according to your model's input
    try:
        year = request.form.get('year', type=int)  # Example feature, replace with actual features
        
        
        # Assuming your model expects a single feature for prediction, adjust as necessary
        # If your model requires preprocessing or expects more features, adjust this accordingly
        features = np.array([[year]])  # Example to adjust
        
        # Make prediction
        prediction = model.predict(features)
        
        # Format the prediction for displaying purposes
        prediction_text = f'The predicted value is: {prediction[0]}'
    except Exception as e:
        prediction_text = f'Error making prediction: {e}'

    # Render the results page with the prediction result
    return render_template('results.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
