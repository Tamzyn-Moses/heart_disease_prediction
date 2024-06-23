import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

conn = sqlite3.connect('heart_disease.db')
df = pd.read_sql_query("SELECT * FROM heart_data", conn)
conn.close()

df = df['age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'].str.split(';', expand=True)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

df[cat_cols] = df[cat_cols].apply(pd.Categorical)
df[num_col] = df[num_col].apply(pd.to_numeric)

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])

X = df.drop('target', axis=1)
y = df['target'].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Accuracy: {accuracy:.2f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

joblib.dump(best_model, 'best_model.pkl')

print(f"Moodel {best_model.__class__.__name__} saved to disk.")
