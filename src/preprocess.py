# src/preprocess.py
import pandas as pd
import yaml
import os
import shutil

from utils.common import categorizeColumns,  detectInvalidValues, handlingEmptyValues, readEnv, readFolder

def preprocess():
    dataset,target, _,_,_,_,_,_= readEnv()
    outputFile = "data/clean_data.csv"
    
    folder = 'models'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Cleared existing contents in '{folder}' folder.")
    os.makedirs(folder, exist_ok=True)

    folder = 'data'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Cleared existing contents in '{folder}' folder.")
    os.makedirs(folder, exist_ok=True)
    
    rootPath = os.getcwd()
    models_list = readFolder("datasets","csv")

    df = pd.DataFrame([])
    for model_name in models_list:   
        model_fullPath = os.path.join(os.getcwd(),model_name)
        df_aux = pd.read_csv(model_fullPath)      
        df_aux = df_aux[df_aux["category"].notnull()]
        df = pd.concat([df,df_aux])

    os.chdir(rootPath)   # Restore the base root path   

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
