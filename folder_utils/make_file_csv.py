import pandas as pd
import os

df = pd.read_csv('datasets/datasets.csv')

df = df.drop_duplicates()
n = 60
df_filtered = df[df['TaxiID'] <= n]
df_filtered = df_filtered[df_filtered['Longitude'] <= 117.4]
df_filtered = df_filtered[df_filtered['Latitude'] <= 41]
df_filtered = df_filtered[df_filtered['Longitude'] >= 115.4]
df_filtered = df_filtered[df_filtered['Latitude'] >= 39.4]
df_filtered = df_filtered[df_filtered['TimeStamp'].str.startswith(('2008-02-03', '2008-02-04'))]

output_dir = '/home/minsun/Documents/Implementing Project/CS313.P23.G9/datasets'  

os.makedirs(output_dir, exist_ok=True)

output_file_path = os.path.join(output_dir, f'taxi_data_{n}.csv')

df_filtered.to_csv(output_file_path, index=False)

print(f"Saved at: {output_file_path}")
