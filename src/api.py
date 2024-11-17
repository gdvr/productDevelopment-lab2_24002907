from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
from typing import Dict, Type, Union, List
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

numericas = params['continuas'] + params['discretas'] 
categoricas = params['categoricas']   

inputFile = 'data/top_features.csv'
df_features =  pd.read_csv(inputFile)

df =  pd.read_csv("data/results.csv")
best_model_name = df.iloc[0]["model"]

app = FastAPI()

model = joblib.load(f"models/{best_model_name}")
preprocessor = joblib.load('models/preprocessor.pkl')


# Dynamically create a Pydantic model based on the CSV columns
def create_dynamic_model() -> Type:
    fields: Dict[str, Union[float, str]] = {col: (float, ...) for col in numericas}
    fields.update({col: (str, ...) for col in categoricas})
    print(fields)
    return create_model("PredictionRequest", **fields)

# Create the dynamic model
PredictionRequest = create_dynamic_model()

@app.post("/predict")
def predict(data:  List[PredictionRequest]):
    print(data)
    try: 
        predictions = []
        features = df_features['feature'].values
        for sample in data:
            data_dict = sample.dict()    
            input_data = pd.DataFrame([data_dict])
            X_transformed_df = preprocess_sample(input_data,features)
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_transformed_df)[0] 
                class_names = model.classes_  
                probability_dict = {str(class_name): float(prob) for class_name, prob in zip(class_names, proba)}               
            else:
                probability_dict = None  # If no probability support
            
            predictions.append(probability_dict) 

        response = {"predictions": predictions}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    

@app.get("/")
def healthCheck():
    return "API online"
        

def preprocess_sample(input_data: pd.DataFrame, features) -> pd.DataFrame:
    X_transformed = preprocessor.transform(input_data)
    num_features = preprocessor.transformers_[0][2]
    if(len(preprocessor.transformers_) > 1 and len(preprocessor.transformers_[1]) > 0):
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categoricas)
        all_feature_names = list(num_features) + list(cat_features)    
    else:
        all_feature_names = list(num_features)
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    return X_transformed_df[features]