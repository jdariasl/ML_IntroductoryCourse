
# Introducción

A continuación vamos a explorar acerca cómo desplegar fácilmente tus modelos en sklearn como un endpoint de API usando el framework fastAPI. Se recomienda estar familiarizado con sklearn y lo básico de APIs. 

## Antes de iniciar:


Aquí hay algunos recursos que pueden ayudarte a familiarizarte:


* Documentación de [fastAPI](https://fastapi.tiangolo.com/)
* Introducción a APIs [APIs](https://www.freecodecamp.org/news/what-is-an-api-in-english-please-b880a3214a82/)

Y se deja como alternativa el uso de docker para correr la API, para ello se debe tener previamente instalado docker.


# Estructura del proyecto

En este tutorial se explicarán como desplegar un modelo de la forma sencilla y simple, sin embargo, si desean realizar una estructura más completa para un sistema más robusto pueden entrar aquí [link](https://github.com/eightBEC/fastapi-ml-skeleton/tree/master/fastapi_skeleton)

En este proyecto vamos a crear dos carpetas de la siguiente forma:

> model
> >  model_building.ipynb
>
> >  api_testing.ipynb
>
> > model_neigh.joblib

> api
> > main.py

> Dockerfile


* `model/model_building.ipynb` -  Donde vamos a entrenar y guardar el modelo en un archivo con extensión 'joblib' 
* `model/api_testing.ipynb` - Donde vamos a probar la API, una vez se despliegue el modelo.
* `model/model_neigh.joblib` - Modelo guardado.
* `app/main.py` - ¡Aquí está la magia de la API!
* `Dockerfile` - Instancia de docker


# Construir el modelo

Se puede usar cualquier notebook que está disponible en el repositorio del [curso](https://github.com/jdariasl/ML_IntroductoryCourse/tree/master/Labs), en este caso vamos a usar un ejemplo sencillo para mejor ilustración, se va a entrenar un K vecinos más cercano con el dataset iris, esto se hará en el archivo `model_building.ipynb`

Estas son las librerías que usaremos:
```
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

import numpy as np
```

Ahora separamos el conjunto de entrenamiento y test, esto es con el fin de probar la API con esta última partición. El 70% es para entrenamiento y el resto para test.

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


* Para probar la API requerimos de datos de prueba, por esta razón, vamos en un archivo plano la partición `X_test`, en este caso será en .csv

`np.savetxt('X_test.csv', X_test, delimiter=',')`

# Contruyendo la API

Vamos a crear la API, dentro de la carpeta `/app` usaremos el archivo `main.py`

1. Importar librerías:

```
from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel
```

* FastAPI, framework que nos va a permitir desplegar el modelo
* Joblib, librería para cargar el modelo guardado
* BaseModel, objeto que define la entrada al endpoint

2. Definir el objeto de entrada

```
class Iris(BaseModel):

    data: List[conlist(float, min_items=4, max_items=4)]
```

3. Cargar el modelo entreanado, recuerda llamarlo igual. Además, se crea la instacia de FastAPI con su titulo y descripción.
```
clf = load('model_neigh.joblib')

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")

```

4. Implementar método de `get_prediction()`, este se encargará de usar los métodos de sklearn para hacer las predicciones, este caso nos devolverá la etiqueta y la distribuición de probabilidad de las 3 clases.

```
def get_prediction(data: List):
    prediction = clf.predict(data).tolist()
    log_proba = clf.predict_proba(data).tolist()
    return {"prediction": prediction,
            "pred_proba": log_proba}
```  

Debemos convertir las predicciones que están como `numpy.array()` en un lista, por esta razón, hacemos uso del método `tolist()`

5. Inicializar el endpoint por método POST

```
@app.post('/predict', tags=["predictions"])
async def predict(iris: Iris):
    data = dict(iris)['data']
    result = get_prediction(data)
    return result
```

Como se observa debe recibir un objeto tipo Iris, tal cual como se definió en el paso 2, para luego llamar el método hecho previamente y se encargue de entregarnos el resultado.
A este punto, ya contamos con una API diseñada para hacer predicciones con un modelo que ya está entrenado. 

6. [Opcional] Construir imagen de docker

En el archivo llamado Dockerfile, este contiene la imagen oficial de FastAPI, en este [link](https://fastapi.tiangolo.com/deployment/) puedes encontrar más información.
Adicionalmente, debemos instalar dos librerías en el contenedor (joblib, scikit-learn).

```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install joblib scikit-learn

COPY ./model/ /model/

COPY ./app /app
```

Ahora bien, vamos a construir la imagen:
`docker build -t myapi .`


Y correrla:
`docker run -d --name myapicontainer -p 80:80 myapi`

¡Y eso es todo! Ya puede realizar solicitudes en `http://localhost/predict`.


# Probando la API

Se puede realizar de dos formas, la primera es en un notebook nuevo para esto debemos guardar los datos en un texto plano para cargalos y esto simula que son los datos nuevos para probarlos, para esto utilizaremos la partición inicial llamada X_test.
En el notebook `model_building.ipynb` vamos a guardar en un .csv las muestras X_test, para luego cargarlos en el notebook `api_testing.ipynb`.

Para iniciar importaremos la librería `requests` para hacer la petición POST a la API, y que nunca falte nuestro aliado numpy

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



