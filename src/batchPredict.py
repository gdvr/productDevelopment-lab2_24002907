import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import yaml
import time
import schedule
from utils.common import readEnv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

_,_,_,_,_,inputFolder,outputFolder,_= readEnv()

df =  pd.read_csv("data/results.csv")
best_model_name = df.iloc[0]["model"]

input_folder = Path(inputFolder)
output_folder = Path(outputFolder)
output_folder.mkdir(parents=True, exist_ok=True)

model = joblib.load(f"models/{best_model_name}")
preprocessor = joblib.load('models/preprocessor.pkl')

inputFile = 'data/top_features.csv'
df_features =  pd.read_csv(inputFile)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

numericas = params['continuas'] + params['discretas'] 
categoricas = params['categoricas']   


def process_parquet_files():
    logger.info("Processing parquet files...")
    parquet_files = list(input_folder.glob("*.parquet"))

    if (len(parquet_files)== 0):
         logger.info(f"Not any file on: {input_folder}")
    
    features = df_features['feature'].values
    for file_path in parquet_files:
        try:
            data = pd.read_parquet(file_path)
            logger.info(f"Processing file: {file_path}")

            if data.empty:
                logger.info(f"No data in file: {file_path}")
                continue

            
            X_transformed = preprocess_sample(data, features)
            predictions = model.predict_proba(X_transformed)
            prediction_df = pd.DataFrame(predictions, columns=model.classes_)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_folder / f"predictions_{file_path.stem}_{timestamp}.csv"
            prediction_df.to_csv(output_file, index=False)

            logger.info(f"Predictions saved to: {output_file}")

            file_path.unlink()  #Delete proccesed files
        except Exception as e:
            logger.info(f"Failed to process {file_path}: {e}")

def preprocess_sample(input_data: pd.DataFrame, features) -> pd.DataFrame:
    X_transformed = preprocessor.transform(input_data)
    num_features = preprocessor.transformers_[0][2]
    if(len(preprocessor.transformers_) > 1 and len(preprocessor.transformers_[1]) > 0):
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categoricas)
        all_feature_names = list(num_features) + list(cat_features)    
    else:
        all_feature_names = list(num_features)
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    return X_transformed_df[features]


#Run every minute
schedule.every(1).minute.do(process_parquet_files)

if __name__ == "__main__":
    logger.info("Starting batch job...")
    while True:
        schedule.run_pending()
        time.sleep(1)    