# src/preprocess.py
import pandas as pd
import sys
import optuna
import yaml

from utils.common import  chooseBestHiperparameters, gb_objective, rf_objective

def searchHiperparameters(input_file,target):   
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

    random_state = params['train']['RANDOM_STATE'] 

    rf_study = optuna.create_study(direction="minimize")
    rf_study.optimize(lambda trial: rf_objective(trial, X_train, y_train, X_val, y_val, random_state), n_trials=50)
    print("Best Random Forest parameters:", rf_study.best_params)

    gb_study = optuna.create_study(direction="minimize")
    gb_study.optimize(lambda trial: gb_objective(trial, X_train, y_train, X_val, y_val, random_state), n_trials=50)
    print("Best Gradient Boosting parameters:", gb_study.best_params)

    params["optuna"] = {
        "RandomForest": rf_study.best_params,
        "GradientBoosting":  gb_study.best_params
    }

    
    models_best_params = chooseBestHiperparameters(X_train,y_train,params['train']['CV'],random_state)
    params["optimization"] = models_best_params  

    with open("params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    input_file = sys.argv[1]
    params_file = sys.argv[2]

    with open(params_file) as f:
        params = yaml.safe_load(f)    

    target = params['preprocessing']['target']

    searchHiperparameters(input_file,target)