from flask import Flask, request, render_template
from flask_pymongo import PyMongo
import pandas as pd
import joblib
import dill
import numpy as np
import pandas as pd
import logging
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler

app = Flask(__name__)
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/history')
mongo = PyMongo(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load model and preprocessing objects
model = joblib.load('best_model.pkl')

with open("categorize_bmi.dill", "rb") as f:
    categorize_bmi = dill.load(f)

with open("set_insulin.dill", "rb") as f:
    set_insulin = dill.load(f)

with open("ohe_encoder.dill", "rb") as f:
    ohe_encoder = dill.load(f)

with open("robust_encoder.dill", "rb") as f:
    robust_encoder = dill.load(f)

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/history')
def history():
    records = mongo.db.records.find()
    return render_template('history.html', records=records)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        logging.info("Form data received: %s", request.form)

        try:
            gender = request.form.get('gender', 'unspecified')  # Optional: Store for reference

            pregnancies = safe_float(request.form.get('pregnancies'))
            glucose = safe_float(request.form.get('glucose'))
            blood_pressure = safe_float(request.form.get('bloodPressure'))
            skin_thickness = safe_float(request.form.get('skinThickness'))
            insulin = safe_float(request.form.get('insulin'))
            bmi = safe_float(request.form.get('bmi'))
            diabetes_pedigree_function = safe_float(request.form.get('diabetesPedigreeFunction'))
            age = safe_float(request.form.get('age'))

            if glucose <= 0 or bmi <= 0 or age <= 0:
                return render_template('form.html', prediction_text="Please enter valid positive values.")

            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree_function],
                'Age': [age]
            })

            input_data = categorize_bmi(input_data)
            input_data = set_insulin(input_data)

            input_encoded = ohe_encoder.transform(input_data[['NewBMI', 'NewInsulinScore']])
            numerical_columns = [col for col in input_data.columns if col not in ['NewBMI', 'NewInsulinScore']]
            input_scaled = robust_encoder.transform(input_data[numerical_columns])

            final_input = np.concatenate([input_scaled, input_encoded], axis=1)
            prediction = model.predict(final_input)

            result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

            # Store prediction in DB
            insert_result = mongo.db.records.insert_one({
                'Gender': gender,
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree_function,
                'Age': age,
                'Prediction': result
            })
            logging.info("Data inserted into MongoDB with ID: %s", insert_result.inserted_id)

        except Exception as e:
            logging.error("Error occurred: %s", e)
            return render_template('form.html', prediction_text="An error occurred while processing your request.")

        return render_template('form.html', prediction_text=f"The person is {result}")

if __name__ == "__main__":
    app.run(debug=True)
