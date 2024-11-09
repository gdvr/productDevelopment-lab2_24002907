# src/preprocess.py
import pandas as pd
import sys

from utils.common import categorizeColumns,  detectInvalidValues, handlingEmptyValues, readEnv

def preprocess(input_file, output_file, features, target):
    readEnv()
    df = pd.read_csv(input_file)
    df = df.dropna()
    columns = features + [target]
    df = df[columns]

    continuas, discretas, categoricas = categorizeColumns(df[features])
    detectInvalidValues(df[columns])
    handlingEmptyValues(df[columns].copy(),continuas + discretas)

    params['continuas'] = continuas
    params['discretas'] = discretas
    params['categoricas'] = categoricas

    with open("params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Datos guardados en {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    
    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)
    
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)
