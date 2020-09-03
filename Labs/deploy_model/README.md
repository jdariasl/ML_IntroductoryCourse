# Introducción

Vamos a explorar acerca de cómo desplegar fácilmente tus modelos en sklearn usando el framework fastAPI. Se recomienda estar familiarizado con sklearn, lo básico de APIs y fastAPI. 

## Antes de iniciar:


Aquí hay algunos recursos que pueden ayudarte a familiarizar:


* Documentación de [fastAPI](https://fastapi.tiangolo.com/)
* Introducción a [APIs](https://www.freecodecamp.org/news/what-is-an-api-in-english-please-b880a3214a82/)

Y además el uso de docker que contiene la imagen con fastAPI y demás recursos, para ello se debe tener previamente instalado docker.

# Estructura del proyecto

En este tutorial se explicarán como desplegar un modelo de la más forma sencilla y simple, sin embargo, si desean realizar una estructura más completa para un sistema más robusto pueden entrar aquí [link](https://github.com/eightBEC/fastapi-ml-skeleton/tree/master/fastapi_skeleton) y organizar la estructura siguiendo dicho orden.

En este proyecto se tiene dos carpetas distribuidas de la siguiente forma:

> model
> > model_building.ipynb
>
> > api_testing.ipynb
>
> > model_neigh.joblib

> api
> > main.py

> Dockerfile


* `model/model_building.ipynb` -  Notebook para entrenar y guardar el modelo en un archivo con extensión 'joblib' 
* `model/api_testing.ipynb` - Notebook para probar la API, una vez, se despliegue el modelo.
* `model/model_neigh.joblib` - Modelo entrenado.
* `app/main.py` - ¡Aquí está la magia de la API!
* `Dockerfile` - Instancia de docker


# Construir el modelo

Se puede usar cualquier notebook disponible en el repositorio del [curso](https://github.com/jdariasl/ML_IntroductoryCourse/tree/master/Labs) para trabajar entrenar el modelo que se desea, en este caso vamos a usar un ejemplo sencillo para mejor ilustración. Se va a entrenar un K vecinos más cercano con el dataset iris, este dataset cuenta con 150 muestras y 4 carácterisitcas, lo haremos dentro del archivo `model_building.ipynb`.

Estas son las librerías que usaremos:
```
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

import numpy as np
```

Ahora separamos el conjunto de entrenamiento y test,  esto es con el fin de probar la API con esta última partición para obtener sus predicciones. El 70% es para entrenamiento y el resto para test.

```
(X, y) = datasets.load_iris(return_X_y=True)
N = np.size(X,0)
split = int(N*0.7)
X_train, y_train = X[0: split], y[0: split]
X_test, y_test = X[split:N], y[split:N]
```

A continuación, vamos a entrenar el modelo con 3 vecinos:

```
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
```

Y una vez entrenado usaremos la librería joblib para guardar el modelo:

```
from joblib import dump, load
dump(neigh, 'model_neigh.joblib')
```

Ahora bien, ya tenemos entrenado y guardado el modelo ya podemos construir el endpoint.

* Para probar la API requerimos de datos de prueba, por esta razón, vamos en un archivo plano la partición `X_test`, vamos a guardarlo como un .csv

`np.savetxt('X_test.csv', X_test, delimiter=',')`

# Contruyendo la API

Vamos a crear la API, dentro de la carpeta `/app` está el archivo `main.py`, en este archivo es donde vamos a trabajar.

1. Importar librerías:

```
from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel, conlist
from typing import List
```


* FastAPI, framework que nos va a permitir desplegar el modelo
* Joblib, librería para cargar el modelo guardado
* BaseModel, objeto que define la entrada al endpoint

2. Definir el objeto de entrada

```
class Iris(BaseModel):

    data: List[conlist(float, min_items=4, max_items=4)]
```
En el variable `data` definimos el objeto de entrada como listas de listas con máximo 4 items, esto quiere decir que es una matriz de tipo float donde cada muestra tiene máximo 4 valores, porque son 4 carácteristicas. [Aquí](https://github.com/eightBEC/fastapi-ml-skeleton/blob/f4f1e6e378093786f96d9db82ad0473645a0c7e4/fastapi_skeleton/models/payload.py) hay un ejemplo de otra forma de definir el objeto de entrada.

3. Cargando el modelo entrenado, recuerda llamarlo igual como fue guardado. Además, se crea la instacia de FastAPI con su titulo y descripción.
```
clf = load('model_neigh.joblib')

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")

```

4. Implementar método de `get_prediction()`, este se encargará de usar los métodos de sklearn para hacer las predicciones, este caso nos devolverá la etiqueta de las 3 diferentes clases que tiene el problema con `predict(data)` y con `predict_proba(data)`la distribuición de probabilidad de las 3 clases.

```
def get_prediction(data: List):
    prediction = clf.predict(data).tolist()
    log_proba = clf.predict_proba(data).tolist()
    return {"prediction": prediction,
            "pred_proba": log_proba}
```  

Cade añadir que debemos convertir las predicciones que como `numpy.array()` en un lista, por esta razón, hacemos uso del método `tolist()`

5. Inicializar el endpoint por método POST

```
@app.post('/predict', tags=["predictions"])
async def predict(iris: Iris):
    data = dict(iris)['data']
    result = get_prediction(data)
    return result
```

Como se observa debe recibir un objeto tipo Iris, tal cual como se definió en el paso 2, para luego llamar el método `get_prediction()` y se encargue de entregarnos el resultado.
A este punto, ya contamos con una API diseñada para hacer predicciones con un modelo que ya está entrenado. 

6. Construir imagen de docker

En el archivo llamado Dockerfile, este contiene la imagen oficial de FastAPI, en este [link](https://fastapi.tiangolo.com/deployment/) puedes encontrar más información.
Adicionalmente, debemos instalar dos librerías en el contenedor (joblib, scikit-learn).

```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install joblib scikit-learn

COPY ./model/ /model/

COPY ./app /app
```

Ahora bien, vamos a construir la imagen:
`docker build -t skl-api .`


Y correrla:
`docker run -d --name apicontainer -p 80:80 skl-api`

¡Y eso es todo! Ya puede realizar solicitudes en `http://localhost/predict`.


# Probando la API
Cargarlos en el notebook `api_testing.ipynb` el archivo X_test.csv, esto simula que son datos nuevos que llegaron de forma externa.

Para iniciar importaremos la librería `requests` para hacer la petición POST, y que nunca falte `numpy` y `pandas`

```
import requests
import numpy as np
import pandas as pd
```

Vamos a cargar .csv de prueba y se enviará como parámetro tipo json, de esta forma enviaremos los datos por una petición POST indicando la URL que está disponible.

```
df = pd.read_csv('/model/X_test.csv')
X_test = df.to_numpy()
data = {"data": X_test.tolist()}
```
Lo guardamos como una lista y lo añadimos como si fuera un diccionario para enviarlo.
Y ahora, con librería `request` hacemos la petición 
```
r = requests.post('http://localhost/predict', json = data)
r.json()

```

Por último, convertimos la respuesta con `r.json()` y así es como debería aparecer:

`{'prediction': [2, 1, 2, 2, 2, 1, 1, 2, 1, 1],
 'pred_proba': [[0.0, 0.0, 1.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.6666666666666666, 0.3333333333333333],
  [0.0, 0.6666666666666666, 0.3333333333333333],
  [0.0, 0.0, 1.0],
  [0.0, 0.6666666666666666, 0.3333333333333333],
  [0.0, 0.6666666666666666, 0.3333333333333333]]}`

Hemos terminado, este es un tutorial básico para desplegar modelos.



