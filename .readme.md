Para correr exitosamente el autoML es necesario algunas consideraciones iniciales:
**Modelos aceptados**
- NaiveBayes
- RandomForest
- GradientBoosting
- SVM
- KNN
* **Nota:** Si se ingresa cualquer otro valor el experimento fallaria.

**DEPLOYMENT_TYPE aceptados**
- batch
- API
* **Nota:** Si se ingresa cualquer otro valor el experimento fallaria.

**Comandos Docker**
- Crear la imagen
``` docker 
 docker build --no-cache  -t lab2 .
```

- Container en modo API
```docker
docker run -v C:\Users\gerda\OneDrive\Documentos\Maestria\productDevelopment-lab2_24002907\datasets:/app/datasets --rm -d --env-file .\.env -p 5000:5000 --name lab2_api lab2:latest
```

- Container en modo Batch
```docker
docker run -v docker run -v C:\Users\gerda\OneDrive\Documentos\Maestria\productDevelopment-lab2_24002907\datasets:/app/datasets -v C:\Users\gerda\OneDrive\Documentos\Maestria\productDevelopment-lab2_24002907\input:/app/input -v C:\Users\gerda\OneDrive\Documentos\Maestria\productDevelopment-lab2_24002907\output:/app/output --rm -d --env-file .\.env --name lab2_batch lab2:latest
```
**Nota:** Dependiendo el modo se debe de adjuntar los volumenes requeridos para poder tener acceso a los archivos durante el proceso de ejecucion de la pipeline.

- Api: Se necesita montar el volumen asociado a la variable de entorno DATASET.
- Batch:
	- Se necesita montar el volumen asociado a la variable de entorno DATASET
	- Se necesita montar el volumen asociado a la variable de entorno INPUT_FOLDER
	- Se necesita montar el volumen asociado a la variable de entorno OUTPUT_FOLDER


**Archivo ENV de ejemplo**
```env
DATASET="datasets/WineQT.csv.parquet"
TARGET=quality
MODEL="GradientBoosting"
TRIALS=10
DEPLOYMENT_TYPE=batch
INPUT_FOLDER=input
OUTPUT_FOLDER=output
# Archivo .env para API
PORT=5000
```
**Nota:** Para los valores de tipo texto aceptan que los valores esten encerrados en comillas en vez del texto plano.

## DataSets
### Entrenamiento
Se adiciona una carpeta llamada dataset con 2 archivos estructurados:
- heart.csv.parquet, tiene informacion relacionado a condiciones para sufrir o no un paro cardiaco.
- WineQT.csv.parquet, tiene informacion relacionada a la calidad del vino entre una categoria de 3 hasta 8.

### Prueba en batch
En la misma carpeta llamada dataset se adicionan 2 archivos estructurados para probar la funcion de batch, estos archivo deben ser depositados en la carpeta asociado a la variable de entorno INPUT_FOLDER donde cada minuto estara verificando si existen archivos para procesar, una vez procesada los deposita en la carpeta asociada a la variable de entorno OUTPUT_FOLDER.
- test_HAtack _2.csv.parquet
- testHAtack.csv.parquet

### Prueba en API
Se adjuntan ejemplos para consumir la API dependiendo que archivo hayan seleccionado para su entramiento:
####  Heart Atack
```JSON
[
    {
        "Age": 10,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 152,
        "Cholesterol": 200,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0,
        "ST_Slope": "Up"
    },
     {
        "Age": 49,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 155,
        "Cholesterol": 180,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 1,
        "ST_Slope": "Up"
    }
]
```

#### Wine Quality
```JSON
[
    {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "Id": 0
    }
]
```
