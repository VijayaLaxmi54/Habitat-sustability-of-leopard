import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

df = pd.read_csv('dataset.csv')
X = df.drop('Habitat_Suitability', axis=1)
y = df['Habitat_Suitability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if os.path.exists('models/rf_model.pkl'):
    rf_model = joblib.load('models/rf_model.pkl')
    rf_preds = rf_model.predict(X_test_scaled)
    print(f"Random Forest Expected Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

if os.path.exists('models/maxent_model.pkl'):
    me_model = joblib.load('models/maxent_model.pkl')
    me_preds = me_model.predict(X_test_scaled)
    print(f"MaxEnt Expected Accuracy: {accuracy_score(y_test, me_preds):.4f}")
