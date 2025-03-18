import os
import sqlite3
import pandas as pd
import sys

def load_csv_to_sqlite(csv_dir, db_path):
    try:
        # Connect to (or create) the SQLite database file
        conn = sqlite3.connect(db_path)
        for file in os.listdir(csv_dir):
            if file.endswith(".csv"):
                table_name = os.path.splitext(file)[0]
                file_path = os.path.join(csv_dir, file)
                print(f"Loading {file} into table {table_name}...")
                
                # Read CSV file into a DataFrame
                df = pd.read_csv(file_path, encoding='cp1252')
                
                # Write DataFrame to a SQLite table (replaces table if it already exists)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Loaded {file} into table {table_name}")
        conn.commit()
        conn.close()
        print("Data loading complete.")
    except Exception as e:
        print("Error loading data:", e)
        sys.exit(1)

def main():
    # The container will have the CSV files mounted at /data.
    csv_dir = "/data"
    # The SQLite database file will be created in the current directory.
    db_path = "output/adventure_works.db"
    load_csv_to_sqlite(csv_dir, db_path)

if __name__ == "__main__":
    main()
