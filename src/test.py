import pandas as pd
import joblib

feature_file = 'data/results.csv'    

df =  pd.read_csv(feature_file)
best_model_name = df.iloc[0]["model"]
print("Modelo Ganador:", best_model_name)

inputFile = 'data/top_features.csv'
df_features =  pd.read_csv(inputFile)
df_val =  pd.read_csv("data/X_val.csv")
features = df_features['feature'].values
X_train = df_val.drop(columns=["HeartDisease"], errors='ignore')
X_train = X_train[features]

model = joblib.load(f"models/{best_model_name}")

sample = X_train.iloc[3:4]
print(sample)
# Make predictions
prediction = model.predict(sample)
print("Prediction:", prediction[0])

# Make probability predictions (if the model supports it)
if hasattr(model, "predict_proba"):
    prediction_proba = model.predict_proba(sample)
    print("Prediction Probabilities:", prediction_proba[0])
else:
    print("The model does not support probability predictions.")
