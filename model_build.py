# Installation of required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore")
import dill
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# 1) Exploratory Data Analysis

# Reading the dataset
df = pd.read_csv("diabetes.csv")

# Ensure folders exist for saving plots
folders = ["Histograms", "Boxplots", "Correlation", "Countplots"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# List of numerical columns
numerical_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Create histograms for each numerical column
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"Histograms/{col}_histogram.jpg", format="jpg")
    plt.close()

# Create boxplots for each numerical column
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(f"Boxplots/{col}_boxplot.jpg", format="jpg")
    plt.close()

# Create a correlation matrix
correlation_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig("Correlation/correlation_heatmap.jpg", format="jpg")
plt.close()

# Create a countplot for the 'Outcome' column
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=df, palette='coolwarm')
plt.title('Countplot of Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.savefig("Countplots/outcome_countplot.jpg", format="jpg")
plt.close()

# 2) Data Preprocessing

# Define the transformation function
print(df.isnull().sum().sum(),"null values in entire data")
# Define the BMI categorization function
def categorize_bmi(df):
    """
    Categorizes BMI into ranges and assigns categorical labels.
    """
    NewBMI = pd.Series(
        ["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"],
        dtype="category"
    )
    df["NewBMI"] = None
    df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
    df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
    df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
    df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
    df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
    df.loc[df["BMI"] > 39.9, "NewBMI"] = NewBMI[5]
    return df

# Save the categorize_bmi function
with open("categorize_bmi.dill", "wb") as f:
    dill.dump(categorize_bmi, f)

# Define the insulin categorization function
def set_insulin(df):
    """
    Categorizes insulin values into 'Normal' and 'Abnormal'.
    """
    def categorize(row):
        if row["Insulin"] >= 16 and row["Insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"

    df = df.assign(NewInsulinScore=df.apply(categorize, axis=1))
    return df

# Save the set_insulin function
with open("set_insulin.dill", "wb") as f:
    dill.dump(set_insulin, f)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(drop="first", sparse=False)
with open("onehot_encoder.dill", "wb") as f:
    dill.dump(encoder, f)

# Initialize the RobustScaler
scaler = RobustScaler()
with open("robust_encoder.dill", "wb") as f:
    dill.dump(scaler, f)

# 3) Data Preprocessing for Model Building

# Split the data into training and testing sets
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with open("categorize_bmi.dill", "rb") as f:
    categorize_bmi = dill.load(f)

with open("set_insulin.dill", "rb") as f:
    set_insulin = dill.load(f)

with open("onehot_encoder.dill", "rb") as f:
    ohe_encoder = dill.load(f)

with open("robust_encoder.dill", "rb") as f:
    robust_encoder = dill.load(f)



X_train = categorize_bmi(X_train)
X_test = categorize_bmi(X_test)

X_train = set_insulin(X_train)
X_test = set_insulin(X_test)

# One Hot Encoding
X_train_encoded = ohe_encoder.fit_transform(X_train[["NewBMI", "NewInsulinScore"]])
X_test_encoded = ohe_encoder.transform(X_test[["NewBMI", "NewInsulinScore"]])

encoded_train_df = pd.DataFrame(X_train_encoded, columns=ohe_encoder.get_feature_names_out(["NewBMI", "NewInsulinScore"]))
encoded_test_df = pd.DataFrame(X_test_encoded, columns=ohe_encoder.get_feature_names_out(["NewBMI", "NewInsulinScore"]))


from sklearn.preprocessing import RobustScaler, OneHotEncoder
import pandas as pd
import dill




# One-Hot Encoder
ohe_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Specify columns
categorical_columns = ["NewBMI", "NewInsulinScore"]

# One Hot Encoding
X_train_encoded = ohe_encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = ohe_encoder.transform(X_test[categorical_columns])

# Save the encoder
with open("ohe_encoder.dill", "wb") as f:
    dill.dump(ohe_encoder, f)


# Initialize the RobustScaler
scaler = RobustScaler()

numerical_columns = [col for col in X_train.columns if col not in categorical_columns]


# Scaling numerical columns using RobustScaler
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Save the scaler
with open("robust_encoder.dill", "wb") as f:
    dill.dump(scaler, f)

# Combine encoded and scaled features
X_train_final_before_smote = pd.concat([
    pd.DataFrame(X_train_scaled, columns=numerical_columns, index=X_train.index),
    pd.DataFrame(X_train_encoded, columns=ohe_encoder.get_feature_names_out(categorical_columns), index=X_train.index)
], axis=1)

X_test_final = pd.concat([
    pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index),
    pd.DataFrame(X_test_encoded, columns=ohe_encoder.get_feature_names_out(categorical_columns), index=X_test.index)
], axis=1)

print("X_train sahpe ", X_train_final_before_smote.shape)
print("X_test sahpe ", X_test_final.shape)
y_train_before_smote= y_train.copy()

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)

# Fit SMOTE on the training data and transform it
X_train_final, y_train = smote.fit_resample(X_train_final_before_smote, y_train_before_smote)

# Checking the new shape of the resampled data
print("Resampled X_train shape:", X_train_final.shape)
print("Resampled y_train shape:", y_train.shape)



import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import cross_val_score



# 1) Random Forests Tuning
rf_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_features": [3,  'auto', 'sqrt', 'log2'],
    "min_samples_split": [5,  10,  50],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

rf_model = RandomForestClassifier(random_state = 12345)

# Apply GridSearchCV with cv=3
gs_cv_rf = GridSearchCV(rf_model,
                       rf_params,
                       cv = 3,
                       n_jobs = -1,
                       verbose = 2).fit(X_train_final, y_train)

print("Best parameters for RF:", gs_cv_rf.best_params_)

# 1.1) Final Model Installation for Random Forest
rf_tuned = RandomForestClassifier(**gs_cv_rf.best_params_).fit(X_train_final, y_train)

# 2) LightGBM Tuning
lgbm = LGBMClassifier(random_state = 12345)

lgbm_params = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "n_estimators": [500, 00],
    "max_depth": [3, 5, 8, 10, -1],
    "num_leaves": [31, 50, 100],
    "min_data_in_leaf": [20, 50, 100],
    "boosting_type": ['gbdt', 'dart']
}

# Apply GridSearchCV with cv=3
gs_cv_lgbm = GridSearchCV(lgbm,
                          lgbm_params,
                          cv = 3,
                          n_jobs = -1,
                          verbose = 2).fit(X_train_final, y_train)

print("Best parameters for LightGBM:", gs_cv_lgbm.best_params_)

# 2.1) Final Model Installation for LightGBM
lgbm_tuned = LGBMClassifier(**gs_cv_lgbm.best_params_).fit(X_train_final, y_train)


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# SVC Tuning
svc_params = {
    "C": [0.1, 1, 10, 100],
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree": [2, 3, 4],
    "gamma": ['scale', 'auto']
    
}

svc_model = SVC(random_state=12345)

# Apply GridSearchCV with cv=3
gs_cv_svc = GridSearchCV(svc_model, svc_params, cv=3, n_jobs=-1, verbose=2).fit(X_train_final, y_train)

print("Best parameters for SVC:", gs_cv_svc.best_params_)

# Final Model Installation for SVC
svc_tuned = SVC(**gs_cv_svc.best_params_).fit(X_train_final, y_train)



# KNN Tuning
knn_params = {
    "n_neighbors": [3, 5, 7, 10, 15],
    "weights": ['uniform', 'distance'],
    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    "leaf_size": [30, 40, 50]
}

knn_model = KNeighborsClassifier()


# Apply GridSearchCV with cv=3
gs_cv_knn = GridSearchCV(knn_model, knn_params, cv=3, n_jobs=-1, verbose=2).fit(X_train_final, y_train)

print("Best parameters for KNN:", gs_cv_knn.best_params_)

# Final Model Installation for KNN
knn_tuned = KNeighborsClassifier(**gs_cv_knn.best_params_).fit(X_train_final, y_train)




# 7) Comparison of Final Models
models = [
    ('RF', RandomForestClassifier(random_state=12345, **gs_cv_rf.best_params_)),
    ("LightGBM", LGBMClassifier(random_state=12345, **gs_cv_lgbm.best_params_)),
    ("SVC", SVC(random_state=12345, **gs_cv_svc.best_params_)),
    ("KNN", KNeighborsClassifier(**gs_cv_knn.best_params_))
]

# Evaluate each model
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model, X_train_final, y_train, cv=3, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Save the best model (based on performance or selection)
best_model = models[0][1]  # Assume RF is the best model (you can change this based on the output above)
best_model.fit(X_train_final, y_train)

# Save the model as a .pkl file
joblib.dump(best_model, 'best_model.pkl')

# Optionally, evaluate performance on the test set
print("\nPerformance on the test set:")

for name, model in models:
    model.fit(X_train_final, y_train)
    test_accuracy = model.score(X_test_final, y_test)
    print(f"{name} Test Accuracy: {test_accuracy:.4f}")

# Load the model from the .pkl file (optional)
# loaded_model = joblib.load('best_model.pkl')
# print(f"Loaded model test accuracy: {loaded_model.score(X_test_final, y_test):.4f}")


