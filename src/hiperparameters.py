# src/preprocess.py
import pandas as pd
import sys
import optuna
import yaml
import os

from sklearn.impute import SimpleImputer
from utils.common import  chooseBestHiperparameters, modelToAppyOptimization,  objective, readEnv

def searchHiperparameters(target):   
    inputFile = 'data/top_features.csv'
    _,_,model,trials,_,_,_,_= readEnv()
    os.makedirs('models', exist_ok=True)
    
    df_features =  pd.read_csv(inputFile)

    df_training =  pd.read_csv("data/X_train.csv")
    df_val =  pd.read_csv("data/X_val.csv")

    features = df_features['feature'].values

    X_train = df_training.drop(columns=[target], errors='ignore')
    y_train = df_training[target]
    X_val = df_val.drop(columns=[target], errors='ignore')
    y_val = df_val[target]

    X_train = X_train[features]
    X_val = X_val[features]

    random_state = params['train']['RANDOM_STATE'] 

    modelsToApplyOptiomization = modelToAppyOptimization()
    params["optuna"] = {}
    params["optimization"] = {}

    target_mapping = {
        'Pedido insuficiente': 0,
        'Posible producto eliminando de catalogo': 1,
        'Posible quiebre de stock por pedido insuficiente': 2,
        'Posible venta atípica': 3,
        'Producto sano': 4,
        'inventario negativo': 5,
        'producto nuevo sin movimiento': 6
    }

    #y_numeric = y.map(target_mapping)
    # Replace NaN values with the mean (or another strategy)
    imputer = SimpleImputer(strategy='mean')  # Options: 'mean', 'median', 'most_frequent', 'constant'
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    if model in modelsToApplyOptiomization:
        print(f"start optuna study for: {model}")
        study  = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial,X_train_imputed, y_train.map(target_mapping), X_val_imputed, y_val.map(target_mapping), random_state,model), n_trials=int(trials), n_jobs=-1)
        params["optuna"][model] = study.best_params
        print(f"Best {model} parameters:", study.best_params)

        #models_best_params = chooseBestHiperparameters(X_train_imputed,y_train.map(target_mapping),params['train']['CV'],random_state,model)
        #params["optimization"] = models_best_params  

    with open("params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    params_file = sys.argv[1]

    with open(params_file) as f:
        params = yaml.safe_load(f)    

    target = params['preprocessing']['target']

    searchHiperparameters(target)