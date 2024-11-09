# src/preprocess.py
import pandas as pd
import sys
import yaml

from utils.common import categorizeColumns,  detectInvalidValues, handlingEmptyValues, readEnv

def preprocess():
    dataset,target, _,_,_,_,_,_= readEnv()
    outputFile = "data/clean_data.csv"
    

    df = pd.read_parquet(dataset, engine='pyarrow')      

    df_features = df.drop(columns=[target], errors='ignore')


    features = df_features.columns.to_list()
    columns = features + [target]
    continuas, discretas, categoricas = categorizeColumns(df[features])
    detectInvalidValues(df[columns])
    handlingEmptyValues(df[columns].copy(),continuas + discretas)

    params = {
        "preprocessing": {
            'target': target,
            'features': features
        },
        'train':{
            'TEST_SIZE':0.3,
            'VALIDATE_SIZE':0.2,
            'RANDOM_STATE':2024,
            'CV':5,
            'alpha':0.1,
        }
    }
    params['continuas'] = continuas
    params['discretas'] = discretas
    params['categoricas'] = categoricas

    with open("params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    df.to_csv(outputFile, index=False)
    print(f"Preprocesamiento completado. Datos guardados en {outputFile}")

if __name__ == "__main__":    
    preprocess()
