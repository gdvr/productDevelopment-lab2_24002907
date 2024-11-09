# src/preprocess.py
import pandas as pd
import sys
import optuna
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def preprocess(input_file,output_file, target):        
    data =  pd.read_csv(input_file)
    features = data.columns

    X = data.drop(columns=[target], errors='ignore')
    y = data[target]

    features = X.columns

    pipeline = Pipeline(steps=[
        ('model', RandomForestRegressor(n_estimators=100, random_state=params["train"]["RANDOM_STATE"]))
    ])
    pipeline.fit(X, y)

    # Get feature importances
    importances = pipeline.named_steps['model'].feature_importances_
    all_feature_names =features

    
    importance_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    # Save the top important features
    #top_n_features = params["feature_engineering"]["top_n_features"]
    top_n_features = 10
    top_features = importance_df.head(top_n_features)
    #"data/top_features.csv"
    top_features.to_csv(output_file, index=False)   
    
    print(f"Featuring engineer completado")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    

    preprocess(input_file, output_file, target)