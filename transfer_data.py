import os
import zipfile
import pandas as pd
import tempfile

zip_path = "datasets/datasets.zip"  

with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    files = []
    for root, _, filenames in os.walk(temp_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                files.append(os.path.join(root, filename))

    print(f"Found {len(files)} txt files.")

    dataframes = []

    for file_path in files:
        df = pd.read_csv(file_path, sep=',', names=["TaxiID", "TimeStamp", "Longitude", "Latitude"])
        dataframes.append(df)

    if dataframes:
        df_all = pd.concat(dataframes, ignore_index=True)
    else:
        exit(1)

df_all["TimeStamp"] = pd.to_datetime(df_all["TimeStamp"], errors="coerce")

df_all = df_all.sort_values(by=["TaxiID", "TimeStamp"])

output_path = "datasets/datasets.csv"
df_all.to_csv(output_path, index=False)

print(f"Saved in {output_path}")
