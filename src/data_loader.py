import pandas as pd
from pathlib import Path

def load_data():
    data_path = (
        Path(__file__).resolve().parent.parent/"data"/"student_mental_health.csv"

    )

    df = pd.read_csv(data_path)

    return df

def validate_data(df):

    if df.empty:
        raise ValueError("Dataset is empty!")

    if "Mental_Health_Risk" not in df.columns:
        raise ValueError("Target column is missing!")

    print("Dataset shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())

    return df 
