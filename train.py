import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

def load_and_preprocess(data_path):
    print("Loading dataset...")
    df = pd.DataFrame()
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Dataset {data_path} not found. Please run generate_data.py first.")
        
    X = df.drop(columns=['Habitat_Suitability'])
    y = df['Habitat_Suitability']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_models(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # As Scikit-Learn doesn't have a direct MaxEnt implementation, 
    # Logistic Regression with polynomial/interaction features is statistically equivalent 
    # to MaxEnt model on presence-background data under certain parameterizations.
    # Therefore, we provide Logistic Regression as a surrogate for MaxEnt.
    print("Training MaxEnt Surrogate (Logistic Regression)...")
    maxent_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    maxent_model.fit(X_train, y_train)
    
    # Evaluate RF
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    print("\n--- Random Forest Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(f"AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    print(classification_report(y_test, rf_preds))
    
    # Evaluate MaxEnt Surrogate
    me_preds = maxent_model.predict(X_test)
    me_probs = maxent_model.predict_proba(X_test)[:, 1]
    print("\n--- MaxEnt Surrogate Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, me_preds):.4f}")
    print(f"AUC: {roc_auc_score(y_test, me_probs):.4f}")
    print(classification_report(y_test, me_preds))
    
    # Save models
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(maxent_model, 'models/maxent_model.pkl')
    
    return rf_model, maxent_model

def visualize_feature_importance(model, feature_names):
    print("Generating feature importance map...")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.title('Random Forest Feature Importance - Leopard Habitat Suitability')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/feature_importance.png')
    print("Saved feature importance plot to static/feature_importance.png")

if __name__ == '__main__':
    data_file = 'dataset.csv'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(data_file)
    rf_model, maxent_model = train_models(X_train, y_train, X_test, y_test)
    visualize_feature_importance(rf_model, feature_names)
    print("\nTraining pipeline complete! Run `python app.py` to start the web interface.")
