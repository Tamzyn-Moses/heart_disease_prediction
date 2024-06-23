import sqlite3
import pandas as pd
import os

file_path = 'C:/Users/tamzy/Desktop/Hounours 2024/Block 2/ITDAA4-B12/Assignment/Code/heart.csv'

if not os.path.isfile(file_path):
    print(f"File not found\n")
else:
    df = pd.read_csv(file_path)

    conn = sqlite3.connect('heart_disease.db')
    print("Connected to database\n")

    df.to_sql('heart_data', conn, if_exists='replace', index=False)
    print("Data inserted\n")

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM heart_data LIMIT 5")
    rows = cursor.fetchall()

    col_names = [description[0] for description in cursor.description]
    print("Columns and firat 5 rows:")
    print(col_names)

    for row in rows:
        print(row)

    conn.commit()
    conn.close()
