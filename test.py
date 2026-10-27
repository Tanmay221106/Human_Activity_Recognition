import pandas as pd
import re

data = []

with open("serial.csv", "r") as file:
    for line in file:
        line = line.strip()

        pattern = (
            r'(\d{2}:\d{2}:\d{2}\.\d+)\s+'
            r'ax=([-+]?\d*\.?\d+)\s+'
            r'ay=([-+]?\d*\.?\d+)\s+'
            r'az=([-+]?\d*\.?\d+)\s+'
            r'gx=([-+]?\d*\.?\d+)\s+'
            r'gy=([-+]?\d*\.?\d+)\s+'
            r'gz=([-+]?\d*\.?\d+)'
        )

        match = re.search(pattern, line)

        if match:
            data.append(match.groups())

df = pd.DataFrame(
    data,
    columns=["time", "ax", "ay", "az", "gx", "gy", "gz"]
)

# Convert sensor columns to float
for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
    df[col] = df[col].astype(float)

print(df.head())

# Save cleaned dataset
df.to_csv("clean_sensor_data.csv", index=False)

print("\nDataset saved as clean_sensor_data.csv")