# 🩺 Diabetes Prediction Web Application

A Flask-based web application that predicts whether a person is diabetic or not using a machine learning model. It supports data input, prediction, diet and exercise suggestions, and stores prediction history in MongoDB.

---

## 🚀 Features

- 🧠 Predict diabetes using a pre-trained ML model
- 📝 Input form for health metrics
- 📊 Stores prediction history in MongoDB
- 🍎 Diet and 🏋️ Exercise suggestion pages
- 📁 Clean and modular code structure

---

## 🛠 Tech Stack

- **Backend**: Flask
- **ML Model**: scikit-learn, joblib, dill
- **Database**: MongoDB (via `flask_pymongo`)
- **Frontend**: HTML templates (Jinja2)

---

## 🧪 Model & Preprocessing

- Model: `best_model.pkl`
- Preprocessing tools:
  - `categorize_bmi.dill`: Categorizes BMI levels
  - `set_insulin.dill`: Classifies insulin levels
  - `ohe_encoder.dill`: One-hot encodes categorical features
  - `robust_encoder.dill`: Scales numerical data

---

## 🧾 Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
Flask
flask_pymongo
joblib
dill
pandas
numpy
scikit-learn
```

---

## ⚙️ Environment Setup

Set the MongoDB URI using environment variable (or use default):

```bash
export MONGO_URI="mongodb://localhost:27017/history"
```

---

## 🏃 How to Run

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 📁 File Structure

```
.
├── app.py                    # Main Flask application
├── best_model.pkl            # Pre-trained ML model
├── categorize_bmi.dill       # BMI categorizer
├── set_insulin.dill          # Insulin classifier
├── ohe_encoder.dill          # One-hot encoder
├── robust_encoder.dill       # Robust scaler
├── templates/
│   ├── home.html
│   ├── form.html
│   ├── diet.html
│   ├── exercise.html
│   └── history.html
└── static/                   # Static files (CSS/JS if any)
```

---

## 📌 Routes

| Endpoint       | Description                        |
|----------------|------------------------------------|
| `/`            | Home page                          |
| `/form`        | Form to input health data          |
| `/predict`     | Predicts diabetes from form input  |
| `/diet`        | Diet suggestions page              |
| `/exercise`    | Exercise suggestions page          |
| `/history`     | Shows prediction history           |

---

## 💾 Database

All prediction records are stored in MongoDB under the `records` collection.

---

## 🤖 Prediction Logic

The form captures user health inputs, applies preprocessing (categorization, encoding, scaling), and uses a trained model to predict diabetes risk. Results are stored in the database and shown on the frontend.

---

## 📬 Contact

For queries or suggestions, feel free to reach out!
