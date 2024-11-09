# src/train.py
import pandas as pd
import joblib
import sys
import yaml
import os

from utils.common import createModel

def train(input_file, output_file,  target):
    df_features =  pd.read_csv(input_file)

    df_training =  pd.read_csv("data/X_train.csv")
    df_test =  pd.read_csv("data/X_test.csv")
    df_val =  pd.read_csv("data/X_val.csv")

    features = df_features['feature'].values

    X_train = df_training.drop(columns=[target], errors='ignore')
    y_train = df_training[target]
    X_test = df_test.drop(columns=[target], errors='ignore')
    y_test = df_test[target]
    X_val = df_val.drop(columns=[target], errors='ignore')
    y_val = df_val[target]

    X_train = X_train[features]
    X_test = X_test[features]
    X_val = X_val[features]

    # Entrenar el modelo
    optunaParameters = params['optuna']
    optimizationParameters = params['optimization']

    os.makedirs('models', exist_ok=True)

    models = []

    for modelName in optunaParameters:
        model = createModel(modelName,optunaParameters[modelName])
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{modelName}_optuna.pkl")
        models.append(f"models/{modelName}_optuna.pkl")
        print(f"Modelo entrenado y guardado en model/{modelName}_optuna.pkl")

    for modelName in optimizationParameters:
        model = createModel(modelName,optimizationParameters[modelName])
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{modelName}_optimazed.pkl")
        models.append(f"models/{modelName}_optimazed.pkl")
        print(f"Modelo entrenado y guardado en model/{modelName}_optimazed.pkl")
    

    df_models = pd.DataFrame({"model_name": models})
    df_models.to_csv(output_file, index=False)
    print(f"Modelos entrenados y guardados")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    with open(params_file) as f:
        params = yaml.safe_load(f)    

    target = params['preprocessing']['target']

    train(input_file, output_file, target)
