import pandas as pd
import os

df = pd.read_csv('datasets/datasets.csv')
df = df.drop_duplicates()
exclude_ids = [6, 8, 12, 16, 17, 19, 31, 35, 37, 47, 52, 53, 62]
df = df[~df['TaxiID'].isin(exclude_ids)]

n = 90
df_filtered = df[df['TaxiID'] <= n]
df_filtered = df_filtered[df_filtered['TimeStamp'].str.startswith(('2008-02-03', '2008-02-04'))]

taxi_counts = df_filtered['TaxiID'].value_counts()
valid_ids = taxi_counts[taxi_counts >= 10].index
df_filtered = df_filtered[df_filtered['TaxiID'].isin(valid_ids)]

output_dir = '/home/minsun/Documents/Implementing Project/CS313.P23.G9/datasets'
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, f'filtered_taxi_data.csv')
df_filtered.to_csv(output_file_path, index=False)

print(f"Saved at: {output_file_path}")
