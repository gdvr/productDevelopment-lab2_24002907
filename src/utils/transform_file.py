import pandas as pd

import os
from utils.common import readFolder


models_list = readFolder("datasets","csv")
metrics_output = {}
models_analysis = {}
for model_name in models_list:   
    #We need to concat only the filename because into the readfolder method we change our target path
    print(model_name)
    csvFullPath = os.path.join(os.getcwd(),model_name)
    csv_file_path = csvFullPath # Replace with your CSV file path
    df = pd.read_csv(csv_file_path)
    # Save as Parquet file
    parquet_file_path = f"{model_name}.parquet"  # Specify your Parquet file path
    df.to_parquet(parquet_file_path, engine='pyarrow')  # You can use 'fastparquet' as well

    print(f"File converted and saved as {parquet_file_path}")
