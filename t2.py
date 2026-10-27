import pandas as pd
import re

rows = []

with open("serial.csv", "r") as f:
    for line in f:
        match = re.search(
            r'(\d{2}:\d{2}:\d{2}\.\d+)\s+ax=([-0-9.]+)\s+ay=([-0-9.]+)\s+az=([-0-9.]+)\s+gx=([-0-9.]+)\s+gy=([-0-9.]+)\s+gz=([-0-9.]+)',
            line
        )

        if match:
            rows.append(match.groups())

df = pd.DataFrame(
    rows,
    columns=["time", "ax", "ay", "az", "gx", "gy", "gz"]
)

df.to_csv("clean_sensor_data.csv", index=False)

print(df.head())