import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('heart_disease.db')
df = pd.read_sql_query("SELECT * FROM heart_data", conn)
conn.close()

df = df['age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'].str.split(';', expand=True)

col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = col_names

cat_col = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in cat_col:
    df[col] = df[col].astype('category')

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in num_cols:
    df[col] = pd.to_numeric(df[col])

print(df.head())

plt.figure(figsize=(24, 14))
for i, col in enumerate(cat_col, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=df, x=col, hue='target')
    plt.title(f'Graph of {col}:')
    plt.xlabel(col)
    plt.ylabel('Count')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(title='Heart Disease', loc='upper right')

plt.tight_layout(pad=6.0)
plt.show()

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

plt.figure(figsize=(24, 14))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=col, hue='target', kde=True, element='step')
    plt.title(f'Graph of {col}:')
    plt.xlabel(col)
    plt.ylabel('Density')
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(title='Heart Disease', loc='upper right')

plt.tight_layout(pad=6.0)
plt.show()