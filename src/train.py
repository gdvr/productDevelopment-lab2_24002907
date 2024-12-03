# src/train.py
import pandas as pd
import joblib
import sys
import yaml
import os
from sklearn.impute import SimpleImputer
from utils.common import createModel, modelToAppyOptimization, readEnv

def train(target):
    inputFile = 'data/top_features.csv'
    outputFile = "data/models.csv"
    _,_,modelName,_,_,_,_,_= readEnv()

    df_features =  pd.read_csv(inputFile)
    df_training =  pd.read_csv("data/X_train.csv")
    features = df_features['feature'].values

    X_train = df_training.drop(columns=[target], errors='ignore')
    y_train = df_training[target]
    X_train = X_train[features]

    target_mapping = {
        'Pedido insuficiente': 0,
        'Posible producto eliminando de catalogo': 1,
        'Posible quiebre de stock por pedido insuficiente': 2,
        'Posible venta atípica': 3,
        'Producto sano': 4,
        'inventario negativo': 5,
        'producto nuevo sin movimiento': 6
    }

    # Entrenar el modelo
    optunaParameters = params['optuna']
    optimizationParameters = params['optimization']

    os.makedirs('models', exist_ok=True)

    models = []

    modelsToApplyOptiomization = modelToAppyOptimization()
    params["optuna"] = { }

    imputer = SimpleImputer(strategy='mean')  # Options: 'mean', 'median', 'most_frequent', 'constant'
    X_train_imputed = imputer.fit_transform(X_train)


    if modelName in modelsToApplyOptiomization:
            model = createModel(modelName,optunaParameters[modelName])
            model.fit(X_train_imputed, y_train.map(target_mapping))
            joblib.dump(model, f"models/{modelName}_optuna.pkl")
            models.append(f"models/{modelName}_optuna.pkl")
            print(f"Modelo entrenado y guardado en model/{modelName}_optuna.pkl")

            """
            model = createModel(modelName,optimizationParameters[modelName])
            model.fit(X_train, y_train)
            joblib.dump(model, f"models/{modelName}_optimazed.pkl")
            models.append(f"models/{modelName}_optimazed.pkl")
            print(f"Modelo entrenado y guardado en model/{modelName}_optimazed.pkl")
            """
    else:
        model = createModel(modelName,{})
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{modelName}.pkl")
        models.append(f"models/{modelName}.pkl")
        print(f"Modelo entrenado y guardado en model/{modelName}.pkl")

    df_models = pd.DataFrame({"model_name": models})
    df_models.to_csv(outputFile, index=False)
    print(f"Modelos entrenados y guardados")

if __name__ == "__main__":
    params_file = sys.argv[1]

    with open(params_file) as f:
        params = yaml.safe_load(f)    

    target = params['preprocessing']['target']

    train(target)
