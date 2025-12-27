import pandas as pd

df = pd.read_csv("data/raw/Heart Disease Dataset_kaggle.csv")

df_clean = df.drop_duplicates()

df_clean.reset_index(drop=True, inplace=True)

df_clean.to_csv("data/processed/heart_clean.csv")