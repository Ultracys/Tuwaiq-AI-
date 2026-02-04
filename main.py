import pandas as pd
import numpy as np

cleaning_plan = {
    "age": {"type": "float", "missing": "median_imp", "outliers": "cap_99"},
    "income": {"type": "float", "missing": "median_imp", "outliers": "log1p_cap_99"},
    "city": {"type": "category", "clean": "canonical_city"},
    "signup_time": {"type": "datetime", "tz": "UTC"}
}

def clean_data_project(df_raw):
    df = df_raw.copy()
    
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    
    df["age_missing"] = df["age"].isna().astype(int)
    df["age"] = df["age"].fillna(df["age"].median())
    
    df["income_missing"] = df["income"].isna().astype(int)
    df["income"] = df["income"].fillna(df["income"].median())
    
    df["income"] = df["income"].clip(upper=df["income"].quantile(0.99))
    
    df["city"] = df["city"].str.strip().lower()
    
    df["signup_time"] = df["signup_time"].fillna(df["signup_time"].mode()[0])
    df["signup_time"] = df["signup_time"].dt.tz_localize("UTC", ambiguous='infer')
    
    df["signup_date"] = df["signup_time"].dt.date
    df["signup_hour"] = df["signup_time"].dt.hour
    
    return df

df_raw = pd.read_csv("real_dataset.csv")

df_clean = clean_data_project(df_raw)

print(df_clean.info())
print(df_clean[["signup_time", "signup_date", "signup_hour"]].head())
print(df_clean["signup_time"].dt.tz)
