# src/evaluate.py
import pandas as pd
import joblib
import json
import sys
import os

from utils.common import evaluateModel, readFolder

def evaluate(feature_file, output_file, metrics_file, target):
    df_features =  pd.read_csv(feature_file)
    df_test =  pd.read_csv("data/X_test.csv")

    features = df_features['feature'].values

    X_test = df_test.drop(columns=[target], errors='ignore')
    y_test = df_test[target]
    X_test = X_test[features]

    rootPath = os.getcwd()
    models_list = readFolder("models","pkl")

    metrics_output = {}
    models_analysis = {}
    for model_name in models_list:   
        #We need to concat only the filename because into the readfolder method we change our target path
        model_fullPath = os.path.join(os.getcwd(),model_name)
        model = joblib.load(model_fullPath)
        metrics = evaluateModel(model,X_test,y_test,params['train']['CV'])
        metrics_output[model_name] = metrics
        models_analysis[model_name] = model
    
    os.chdir(rootPath)   # Restore the base root path

    # Guardar métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=4)

    df = pd.DataFrame(metrics_output).T.reset_index()
    df.columns = ["model", "MAE", "MSE", "RMSE", "CV MAE", "R2 Score"]
    df = df.sort_values(by="CV MAE", ascending=True)

    best_model_name = df.iloc[0]["model"]
    print("Modelo Ganador:", best_model_name)

    model = models_analysis[best_model_name]
    if hasattr(model, "get_params"):
        model_params = model.get_params()
        print("Parametros del modelo:", model_params)
    else:
        print("Este modielo no tiene parametros.")

    
    df.to_csv(output_file, index=False)
    print(f"Métricas guardadas en {metrics_file}")
    print(f"Resultados guardadas en {output_file}")
   
    

if __name__ == "__main__":
    feature_input_file = sys.argv[1]
    output_file = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]

    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    target = params['preprocessing']['target']

    evaluate(feature_input_file, output_file, metrics_file, target)
