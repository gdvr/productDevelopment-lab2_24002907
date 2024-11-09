# src/preprocess.py
import pandas as pd
import sys
import yaml

from utils.common import  createPreprocesor, splitValuesForModel

def transform(features, target):
    inputFile = 'data/clean_data.csv'
    outputFile = "data/transformed_data.csv"
    df = pd.read_csv(inputFile)

    numericas = params['continuas'] + params['discretas'] 
    categoricas = params['categoricas']   

    X = df[features]
    y = df[target]

    preprocessor = createPreprocesor(categoricas,numericas)
    X_transformed = preprocessor.fit_transform(X)

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categoricas)
    all_feature_names = list(num_features) + list(cat_features)    
    
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    final_dataset = pd.concat([X_transformed_df, y.reset_index(drop=True)], axis=1)
    final_dataset.to_csv(outputFile, index=False)

    X_train, y_train,X_test,y_test,X_val, y_val = splitValuesForModel(X_transformed,y,params['train']['TEST_SIZE'],params['train']['VALIDATE_SIZE'],params['train']['RANDOM_STATE'])
    X_train = pd.DataFrame(X_train, columns=all_feature_names) if not isinstance(X_train, pd.DataFrame) else X_train
    X_test = pd.DataFrame(X_test, columns=all_feature_names) if not isinstance(X_test, pd.DataFrame) else X_test
    X_val = pd.DataFrame(X_val, columns=all_feature_names) if not isinstance(X_val, pd.DataFrame) else X_val

    # Add the target column back to each dataset
    X_train[target] = y_train.reset_index(drop=True)
    X_test[target] = y_test.reset_index(drop=True)
    X_val[target] = y_val.reset_index(drop=True)

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)

    print(f"Transformacion completado. Datos guardados en {outputFile}")
    
if __name__ == "__main__":
    params_file = sys.argv[1]

    with open(params_file) as f:
        params = yaml.safe_load(f)
    
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    transform(features, target)
