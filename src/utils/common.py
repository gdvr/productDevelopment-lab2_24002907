import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, precision_score,f1_score, accuracy_score, recall_score
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient


models_Def = {
    "NaiveBayes": GaussianNB,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    'SVM': SVC,
    "KNN":KNeighborsClassifier
}

def readFiles(path):
    print(path)
    data = []
    os.chdir(path)
    files = ""
    for file in os.listdir():
        if file.endswith(".xlsx"):
            file_path = f"{path}\{file}"
            if "walmart_semana" in file_path.lower():
                print(f"Processing {file_path}...")
                data.append(read_text_file(file_path))
                print(f"Processed {file_path}")
    if len(data) != 0:
        data = pd.concat(data)
    else:
        print('Error, debe tener el archivo datos en xlsx y archivo ecat en txt')

    return data

def splitValuesForModel(X,y, TEST_SIZE, VALIDATE_SIZE,RANDOM_STATE):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,  y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VALIDATE_SIZE, random_state=RANDOM_STATE)

    print(f"Training set class distribution:\n{X_train.shape}-{y_train.shape}")
    print(f"Validation set class distribution:\n{X_val.shape}-{y_val.shape}")
    print(f"Test set class distribution:\n{X_test.shape}-{y_test.shape}")

    return X_train, y_train,X_test,y_test,X_val, y_val

def categorizeColumns(dataset):
    continuas, discretas, categoricas = __get_variables_scale_type(dataset)
    print(f"# Continuas: {len(continuas)}, values: {', '.join(continuas)}")
    print(f"# Discretas: {len(discretas)}, values: {', '.join(discretas)}")
    print(f"# Categoricas: {len(categoricas)}, values: {', '.join(categoricas)}")

    return continuas, discretas, categoricas

def createPipeline(categoricals, numerics, models):
    preprocessor = createPreprocesor(categoricals,numerics)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', models)
    ])

    return model_pipeline

#Funcion que detecta los valores nulos
def detectInvalidValues(dataset):    
    columnas = dataset.columns
    for col in columnas:        
        porcentaje = dataset[col].isnull().mean()
        if porcentaje > 0:               
            print(f"Percentage of null values for {col}: {porcentaje}%")            
        else:
            print(f"No invalid data for {col}")           

def handlingEmptyValues(dataset,cols):
    print(f"Fill the empty values with mean for cols: {', '.join(cols)}")
    dataset[cols] = dataset[cols].apply(lambda col: col.fillna(col.mean()), axis=0)
    return dataset

#Funcion que permite clasificar las columnas en categoricas, discretas y continuas
def __get_variables_scale_type(dataset):
    columnas = dataset.columns
    categoricas = []
    continuas = []
    discretas = []

    for col in columnas:
        col_type=dataset[col].dtype
        
        if(col_type == 'object' or col_type == 'category'):
            categoricas.append(col)
        elif((col_type =='int64' or col_type =='int32') or (col_type =='float64' or col_type =='float32')):
            n = len(dataset[col].unique())
            if(n > 30):
                continuas.append(col)
            else:
                discretas.append(col)
    
    return continuas, discretas, categoricas     

def objective(trial, X_train, y_train, X_val, y_val, random_state, model_name):
        model = createModel(model_name,{})
        if model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model.set_params(random_state=random_state,n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        elif model_name == 'GradientBoosting':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            max_depth = trial.suggest_int('max_depth', 3, 7)
            model.set_params(random_state=random_state,n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        elif model_name == 'SVM':
            C = trial.suggest_float('C', 0.1, 1, log=True)  # Replaces suggest_loguniform
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model.set_params(random_state=random_state,C=C, kernel=kernel, gamma=gamma)
        elif model_name == 'KNN':
            n_neighbors = trial.suggest_int('n_neighbors', 3, 7)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
            model.set_params(n_neighbors=n_neighbors, weights=weights, metric=metric)

        model.fit(X_train, y_train)
        score = accuracy_score(y_val, model.predict(X_val))
        return score


def hyperparameter_search(model, param_grid, X_train,y_train, cv, search_type):
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=cv, scoring='accuracy', n_jobs=-1)
    
    search.fit(X_train, y_train)
    return search

def chooseBestHiperparameters(X_train,y_train, cv, random_state, modelToApply):
    models_and_params = {
        'RandomForest': (RandomForestClassifier(random_state=random_state),{
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }),
        'GradientBoosting': (GradientBoostingClassifier(random_state=random_state),{
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        'SVM': (SVC(random_state=random_state),{
            'C': [0.1, 0.5, 1],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }),
        'KNN':(KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        })
    }

    best_models = {}
    best_configs = {}
    for mode in ['grid','random']:
        for model_name, (model, param_grid) in models_and_params.items():
            if(model_name ==modelToApply):
                print(f"Running search for {model_name} and {mode} search...")
                search = hyperparameter_search(model, param_grid, X_train,y_train,cv,mode)
                best_model, best_score = search.best_estimator_, search.best_score_
                best_models[model_name] = (best_model, best_score)
                best_configs[model_name] = search.best_params_
                print(f"{model_name} best score: {best_score:.4f}")

    # Compare models and select the best one
    best_model_name = max(best_models, key=lambda k: best_models[k][1])
    best_model, best_score = best_models[best_model_name]

    joblib.dump(best_model, f"models/bestModel_{best_model_name}.pkl")

    print(f"\nBest model: {best_model_name} with score: {best_score:.4f}")
    print(f"Best model details:\n{best_model}")

    return best_configs

def createPreprocesor(categoricals, numerics):
    one_hot_encoder = OneHotEncoder()

    transformers=[]

    if(len(numerics) > 0):
        transformers.append(('num', 'passthrough', numerics))
    if(len(categoricals)> 0):
        transformers.append(('cat', one_hot_encoder, categoricals))

    
    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return preprocessor

def createModel(model_name, params):
    model = models_Def[model_name](**params)
    return model

def evaluateModel(model, x, y, cv):
    if is_classifier(model):
        y_predict = model.predict(x)
        accuracy = accuracy_score(y, y_predict)
        precision = precision_score(y, y_predict, average='weighted')
        recall = recall_score(y, y_predict, average='weighted')
        f1 = f1_score(y, y_predict, average='weighted')
        scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
        
        return {
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1Score': round(f1, 4),
            'CV Accuracy': round(np.mean(scores), 4)
        }
    elif is_regressor(model):
        y_predict = model.predict(x)
        mae = mean_absolute_error(y, y_predict)
        mse = mean_squared_error(y, y_predict)
        rmse = np.sqrt(mse)
        scores = cross_val_score(model, x,y, cv=cv, scoring='neg_mean_absolute_error')
        r2 = r2_score(y, y_predict)

        return {
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'CV MAE': round(-np.mean(scores), 2),
            'R2 Score': round(r2, 2)
        }
    else:
        raise ValueError("Model type not supported. Please provide a classification or regression model.")

def readFolder(path, extension):
    data = []
    os.chdir(path)
    for file in os.listdir():
        if file.endswith(f".{extension}"):
            data.append(file)
    return data

def readEnv():
    load_dotenv()
    dataset = os.getenv("DATASET").strip('"').strip("'")
    target = os.getenv("TARGET").strip('"').strip("'")
    model = os.getenv("MODEL").strip('"').strip("'")
    trials = os.getenv("TRIALS").strip('"').strip("'")
    deploymentType = os.getenv("DEPLOYMENT_TYPE").strip('"').strip("'")
    inputFolder = os.getenv("INPUT_FOLDER").strip('"').strip("'")
    outputFolder = os.getenv("OUTPUT_FOLDER").strip('"').strip("'")   
    port = os.getenv("PORT").strip('"').strip("'")

    print("ENV values:")
    print(dataset,target, model,trials,deploymentType,inputFolder,outputFolder,port, sep=',')

    return dataset,target, model,trials,deploymentType,inputFolder,outputFolder,port
    
def modelToAppyOptimization():
    return [
        "RandomForest",
        "GradientBoosting",
        'SVM',
        "KNN"
    ]

def calculateDF(df_test):
    df_test = df_test.sort_values(by=['storeId', 'productId', 'Fecha']).reset_index(drop=True)
    
    df_test = calculate_last_4_weeks_avg_optimized(df_test)    
    print("Execute calculate_last_4_weeks_avg_optimized")
    df_test = calculate_oldest_sale_date_vectorized(df_test)
    print("Execute calculate_oldest_sale_date_vectorized")
    df_test = totalPresence(df_test)
    print("Execute totalPresence")
    df_test = define_rotation_type(df_test)
    print("Execute define_rotation_type")
    df_test = calculate_percentage_diff_and_condition(df_test)
    print("Execute calculate_percentage_diff_and_condition")
    df_test = calculate_remaining_days_and_broke(df_test)
    print("Execute calculate_remaining_days_and_broke")
    df_test = checkSales(df_test)
    print("Execute checkSales")
    df_test['category'] = df_test.apply(assign_category, axis=1)

    return df_test



def preproccingDF(data):
    df_test = data.copy()    
    catalogDescription = pd.read_excel("products.xlsx")  
    storeDescription = pd.read_excel("stores.xlsx")  
    categoryDescription = pd.read_excel("categories.xlsx")
    catalogDescription['productId'] = (catalogDescription['productId']).astype(str)  
    catalogDescription['OriginalBarCode'] = (catalogDescription['OriginalBarCode']).astype(str)  
    catalogDescription['OriginalBarCode'] = catalogDescription['OriginalBarCode'].apply(lambda x: x.rstrip(".0") if x.endswith(".0") else x)
    print(catalogDescription.dtypes)
    print(data.dtypes)
    
    joined_df = pd.merge(df_test,catalogDescription,how='left',left_on='Código Barra CP',right_on='OriginalBarCode')
    joined_df = pd.merge(joined_df,storeDescription,how='left',left_on='Store Nbr',right_on='OriginalStoreCode')
    joined_df = pd.merge(joined_df,categoryDescription,how='left',left_on='SubBrand CP',right_on='OriginalCategory')
    joined_df = joined_df[['Promedio Rotación Semanal','Inv On Hand','Inv en Tránsito','Inv preparandose en CD','productDescription','productId','storeId','storeDescription','categoryId','País','Fecha']]
    joined_df = joined_df.rename(columns={"Promedio Rotación Semanal": "Venta dia anterior",
                                "Inv On Hand": "Stock dia actual",
                                "Inv en Tránsito": "Pedido en transito",
                                "Inv preparandose en CD": "Pedido procesandose"                         
                                })
    return cleanDF(joined_df)

def cleanDF(df_test):
    duplicates = df_test.duplicated(subset=['Fecha', 'productId', 'storeId','País'], keep=False)
    
    if duplicates.any():
        print("Duplicate rows found:")
        print(df_test[duplicates].shape)
    
        df_test = (
            df_test.groupby(['Fecha', 'productId', 'storeId','País'], as_index=False)
            .agg({'Venta dia anterior': 'sum', 'Stock dia actual': 'sum'})
        )
        print("Duplicates resolved. Cleaned data:")
        print(df_test.shape)
    return df_test

def calculate_last_4_weeks_avg_optimized(df):
    df['Shifted Venta'] = df.groupby(['storeId', 'productId'])['Venta dia anterior'].shift(1) #Exclude the current date
    df['Rolling Avg'] = (
        df.groupby(['storeId', 'productId'])['Shifted Venta']
        .transform(lambda x: x.rolling(window=4, min_periods=1).mean()) #Average for the 4 previous records
    )
    df['Rolling Count'] = (
        df.groupby(['storeId', 'productId'])['Shifted Venta']
        .transform(lambda x: x.rolling(window=4, min_periods=1).count()) # Count for the rows that has more than 4 records
    )
    
    df['Last 4 Weeks Avg'] = df['Rolling Avg'].where(df['Rolling Count'] >= 4, -1)
    # Drop columns
    df.drop(columns=['Shifted Venta', 'Rolling Avg', 'Rolling Count'], inplace=True)
    return df

def calculate_oldest_sale_date_vectorized(df):
    df = df.sort_values(by=["storeId", "productId", "Fecha"]).reset_index(drop=True)

    sales_mask = df["Venta dia anterior"] > 0

    df["Last Sale Date"] = (
        df[sales_mask]
        .groupby(["storeId", "productId"])["Fecha"]
        .transform(lambda x: x.ffill())
    )
    df["Last Sale Date"] = (
        df.groupby(["storeId", "productId"])["Last Sale Date"]
        .ffill()
    )
    df["Days Since Last Sale"] = (df["Fecha"] - df["Last Sale Date"]).dt.days.fillna(-1).astype(int)

    return df



def totalPresence(df_test):
    df_test['Total Presence'] = (
        df_test[df_test['Stock dia actual'] > 0]
        .groupby(['productId', 'País', 'Fecha'])['storeId']
        .transform('nunique')
        .fillna(0)
        .astype(int)
    )
    
    df_test['Total Presence'] = df_test['Total Presence'].fillna(0).astype(int) # FIll the invalid values
    
    total_stores_per_country = df_test.groupby('País')['storeId'].nunique()
    df_test['total tiendas'] = df_test['País'].map(total_stores_per_country)
    df_test['% global tiendas'] = df_test['Total Presence'] / df_test['total tiendas'] # Calculate the percentage of presence
    
    return df_test

def define_rotation_type(df_test):
    last_dates = (
        df_test.groupby(['storeId', 'productId'])['Fecha']
        .max()
        .reset_index(name='Last Date')
    )
    
    df_test = df_test.merge(last_dates, on=['storeId', 'productId'], how='left')

    last_date_rows = df_test[df_test['Fecha'] == df_test['Last Date']]

    mode_values = (
        last_date_rows.groupby(['storeId', 'productId'])['Days Since Last Sale']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index(name='Mode')
    )
    
    mode_values['Rotation Type'] = mode_values['Mode'].apply(classify_rotation_type)

    df_test = df_test.merge(
        mode_values[['storeId', 'productId', 'Rotation Type']],
        on=['storeId', 'productId'],
        how='left'
    )

    df_test.drop(columns=['Last Date'], inplace=True)

    return df_test


def classify_rotation_type(mode_value):
    if pd.isna(mode_value):
        return 'Unknown'
    elif 0 <= mode_value <= 15:
        return 'A'    
    elif 16 <= mode_value <= 60:
        return 'B'    
    elif mode_value > 60:
        return 'C'    
    else:
        return 'Unknown'


def calculate_percentage_diff_and_condition(df):
    df['Percentage Difference'] = (
        df['Venta dia anterior'] / df['Last 4 Weeks Avg'] - 1
    ).replace([float('inf'), -float('inf')], float('nan'))  # Replace invalid values

    thresholds = {'A': 0.25, 'B': 0.40, 'C': 0.50, 'Unknown': 0.50}
    df['Threshold'] = df['Rotation Type'].map(thresholds)
    df['Condition'] = df['Percentage Difference'] > df['Threshold'].fillna(float('inf')) # Calculate the condition based on percentage of variation of sales and thresholds

    return df

def calculate_remaining_days_and_broke(df):
    df['Remaining Days'] = df['Stock dia actual'] / df['Venta dia anterior'].replace(0, 1)

    thresholds = {
        'A': 4,
        'B': 2,
        'C': 1,
        'Unknown': 1,
    } # Days of availables sale
    
    df['Threshold'] = df['Rotation Type'].map(thresholds).fillna(float('inf'))
    df['Remaining Broke'] = df['Remaining Days'] < df['Threshold'] #Compare if the available days of sales is lower than the day thrseholds
    df.drop(columns=['Threshold'], inplace=True)

    return df


def checkSales(df_test):    
    df_test['cumulative_sales'] = (
        df_test.groupby(['storeId', 'productId'])['Venta dia anterior']
        .cumsum()
        .shift(1, fill_value=0)
    ) # Check if has sale parting of the previous date
    
    df_test['has sales'] = df_test['cumulative_sales'] > 0 # Check if the accumulate is greather than zero
    
    df_test['group_index'] = (
        df_test.groupby(['storeId', 'productId'])
        .cumcount()
    )
    df_test['enough information'] = df_test['group_index'] >= 4 #Check if has more than 4 records of information
    df_test.drop(columns=['cumulative_sales', 'group_index'], inplace=True)
    return df_test

def assign_category(row):
    if row['Stock dia actual'] < 0:
        return "inventario negativo"
    elif not row['has sales'] and row['Venta dia anterior'] == 0:
        return 'producto nuevo sin movimiento'
    elif row['Condition'] and row['Remaining Broke'] and row['Remaining Days'] > 0:
        return "Posible venta atípica"
    elif ((row['Condition'] and row['Remaining Broke']) and row['Remaining Days'] <= 0) or (row['Remaining Broke'] and row['Percentage Difference'] <= 0) :
        return "Posible quiebre de stock por pedido insuficiente"
    elif row['% global tiendas'] <= 0.4:
        return 'Posible producto eliminando de catalogo'
    elif not row['Condition'] and row['Remaining Broke'] and row['Remaining Days'] > 0:
        return 'Pedido insuficiente'
    elif row['Stock dia actual'] == 0:
        if row['Pedido en transito'] > 0:
            return 'Pedido pendiente de entregar'        
        elif row['Pedido procesandose'] > 0:
            return 'Pedido realizado tardiamente'
        else:
            return 'Producto con quiebre de stock'
    elif not row['Remaining Broke']:
        return "Producto sano"
    else:
        return None  # For cases that dont apply a correct assign
    
def readResults():
    fields = ['Fecha', 'productId', 'storeId', 'País', 'Venta dia anterior',
       'Stock dia actual', 'Last 4 Weeks Avg', 'Last Sale Date',
       'Days Since Last Sale', 'Total Presence', 'total tiendas',
        '% global tiendas', 'Percentage Difference',
       'Condition', 'Remaining Days', 'Remaining Broke', 'has sales',
       'enough information', 'category']

    client = MongoClient("mongodb://localhost:27017/") 
    db = client["retail_classificator"]
    collection = db["data"]

    data = list(collection.find())
    if len(data) == 0:
        return pd.DataFrame([])

    df_from_db = pd.DataFrame(data)

    datetime_columns = ["Fecha", "Last Sale Date"] 
    for col in datetime_columns:
        if col in df_from_db.columns:
            df_from_db[col] = pd.to_datetime(df_from_db[col], errors="coerce")  
        
    
    print(f"Total of: {df_from_db[fields].shape}")
    return df_from_db[fields]

def insertResults(df_test):
    for col in df_test.select_dtypes(include=["datetime64[ns]"]).columns:
        print(col)
        df_test[col] = df_test[col].apply(lambda x: None if pd.isna(x) else x.isoformat() if isinstance(x, pd.Timestamp) else x)

    data_dict = df_test.to_dict(orient="records")
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["retail_classificator"]
    collection = db["results"]

    if data_dict:  # Only insert if there is data
        collection.insert_many(data_dict)
        print("Data successfully inserted into MongoDB!")
    else:
        print("No data to insert.")