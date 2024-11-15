import os
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import yaml
import time
import schedule


input_folder = Path("input")
output_folder = Path("output/folder")
output_folder.mkdir(parents=True, exist_ok=True)

model = joblib.load("models/RandomForest_optuna.pkl")
preprocessor = joblib.load('models/preprocessor.pkl')

inputFile = 'data/top_features.csv'
df_features =  pd.read_csv(inputFile)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

numericas = params['continuas'] + params['discretas'] 
categoricas = params['categoricas']   


def process_parquet_files():
    parquet_files = list(input_folder.glob("*.parquet"))

    if (len(parquet_files)== 0):
         print(f"Not any file on: {input_folder}")
    
    features = df_features['feature'].values
    for file_path in parquet_files:
        try:
            data = pd.read_parquet(file_path)
            print(f"Processing file: {file_path}")

            if data.empty:
                print(f"No data in file: {file_path}")
                continue

            
            X_transformed = preprocess_sample(data, features)
            predictions = model.predict_proba(X_transformed)
            prediction_df = pd.DataFrame(predictions, columns=model.classes_)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_folder / f"predictions_{file_path.stem}_{timestamp}.csv"
            prediction_df.to_csv(output_file, index=False)

            print(f"Predictions saved to: {output_file}")

            file_path.unlink()  #Delete proccesed files
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

def preprocess_sample(input_data: pd.DataFrame, features) -> pd.DataFrame:
    X_transformed = preprocessor.transform(input_data)
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categoricas)
    all_feature_names = list(num_features) + list(cat_features)    
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    return X_transformed_df[features]


#Run every minute
schedule.every(1).minute.do(process_parquet_files)

print("Scheduler started...")

while True:
    schedule.run_pending()  
    time.sleep(1)           