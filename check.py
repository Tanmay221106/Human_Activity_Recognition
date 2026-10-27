import pandas as pd

FILE_PATH = "D:/Software/PersonalWork/MozekHealthtech/adl/cleaned_sensor_data.csv"

df = pd.read_csv(FILE_PATH)
print(df.shape)
print(df.head())