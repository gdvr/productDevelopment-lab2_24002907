import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import yaml
import time
import schedule
from utils.common import calculateDF, insertResults, preproccingDF, readEnv, readResults
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

fields = ['Fecha', 'productId', 'storeId', 'País', 'Venta dia anterior',
       'Stock dia actual', 'Last 4 Weeks Avg', 'Last Sale Date',
       'Days Since Last Sale', 'Total Presence', 'total tiendas',
        '% global tiendas', 'Percentage Difference',
       'Condition', 'Remaining Days', 'Remaining Broke', 'has sales',
       'enough information', 'category']

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


def process_input_files():
    logger.info("Processing xlsx files...")
    parquet_files = list(input_folder.glob("*.xlsx"))

    if (len(parquet_files)== 0):
         logger.info(f"Not any file on: {input_folder}")
    
    features = df_features['feature'].values
    for file_path in parquet_files:
        try:
            data = pd.read_excel(file_path)
            logger.info(f"Processing file: {file_path}")

            if data.empty:
                logger.info(f"No data in file: {file_path}")
                continue

            
            previous_data = readResults()    
            uniqueDate = np.sort(data['Fecha'].unique()) 
            for date in uniqueDate:
                print(date)
                df_date = data[data['Fecha'] == date]                       
                df_date['Código Barra CP'] = (df_date['Código Barra CP']).astype(str)  
                joined_df = preproccingDF(df_date)
                
                
                if not previous_data.empty:
                    joined_df = pd.concat([joined_df, previous_data], ignore_index=True)
                    
                df_test = joined_df.copy()                
                df_test = calculateDF(df_test)     

                df_test = df_test[df_test['Fecha'] == date]
            
                X_transformed = preprocess_sample(df_test, features)
                print(X_transformed.head)
                predictions = model.predict_proba(X_transformed)

                target_mapping = {
                    'Pedido insuficiente': 0,
                    'Posible producto eliminando de catalogo': 1,
                    'Posible quiebre de stock por pedido insuficiente': 2,
                    'Posible venta atípica': 3,
                    'Producto sano': 4,
                    'inventario negativo': 5,
                    'producto nuevo sin movimiento': 6
                }

                df_test['Last Sale Date'] = df_test['Last Sale Date'].apply(lambda x: None if pd.isna(x) or x == "NaT" else x)  # Replace NaT with None
                df_test['Fecha'] = df_test['Fecha'].apply(lambda x: None if pd.isna(x) else x)                
                
                prediction_df = pd.DataFrame(predictions, columns=model.classes_)
                reverse_target_mapping = {v: k for k, v in target_mapping.items()}
                prediction_df.columns = [reverse_target_mapping[col] for col in prediction_df.columns]

                merged_df = pd.concat([df_test, prediction_df], axis=1)

                insertResults(merged_df)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_folder / f"predictions_{file_path.stem}_{timestamp}.csv"
                merged_df.to_csv(output_file, index=False)
            
            

            logger.info(f"Predictions saved to: {output_file}")

            file_path.unlink()  #Delete proccesed files
        except Exception as e:
            logger.info(f"Failed to process {file_path}: {e}")
            print(e)

def preprocess_sample(input_data: pd.DataFrame, features) -> pd.DataFrame:
    X_transformed = preprocessor.transform(input_data)
    num_features = preprocessor.transformers_[0][2]
    all_feature_names = list(num_features)
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    return X_transformed_df[features]


"""
#Run every minute
schedule.every(1).minute.do(process_parquet_files)

if __name__ == "__main__":
    logger.info("Starting batch job...")
    while True:
        schedule.run_pending()
        time.sleep(1)    
"""

if __name__ == "__main__":
    process_input_files()