import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('heart.csv', delimiter=';')

# Handle missing values
df.replace('NA', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Define features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Combine preprocessing with the classifier (model)
# List of models to evaluate
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Support Vector Machine', SVC(kernel='rbf', random_state=42))
]

# Iterate over models to fit and evaluate
best_model = None
best_accuracy = 0.0

for model_name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Check if current model is the best performing one
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf
        best_model_name = model_name

# Save the best performing model to disk
if best_model:
    joblib.dump(best_model, 'best_model_heart_disease_prediction.joblib')
    print(f"\nBest Model ({best_model_name}) saved to disk.")

