# Importamos os para la creacion y verificacion de carpetas de salida
import os

# Importamos sys para modificar las rutas de importacion de modulos
import sys

# Importamos joblib para serializar y guardar el modelo entrenado y el preprocesador
import joblib

# Importamos numpy para el acotamiento de predicciones
import numpy as np

# Importamos pandas para la manipulacion de datos
import pandas as pd

# Importamos las clases base de scikit-learn para crear un estimador personalizado
from sklearn.base import BaseEstimator, RegressorMixin

# Importamos la regresion Ridge como modelo lineal regularizado
from sklearn.linear_model import Ridge

# Añadimos la carpeta Parte_1 a la ruta para reutilizar sus modulos
sys.path.append("../Parte_1")

# Importamos la particion temporal y el preprocesador oficial de la Parte 1
from preprocesamiento import Preprocesador, partir


# Clase de regresion Ridge acotada en [0, 1] para la LGD
class RegresionRidgeAcotada(BaseEstimator, RegressorMixin):

    # Metodo constructor con parametro de regularizacion L2 (alpha)
    def __init__(self, alpha=1.0):
        # Guardamos el hiperparametro alpha
        self.alpha = alpha
        # Instanciamos el modelo Ridge de scikit-learn
        self.model = Ridge(alpha=self.alpha)

    # Metodo de ajuste sobre los datos de entrenamiento
    def fit(self, X, y):
        # Entrenamos la regresion Ridge con las caracteristicas X y la meta y
        self.model.fit(X, y)
        # Retornamos la instancia entrenada
        return self

    # Metodo de prediccion acotado
    def predict(self, X):
        # Generamos las predicciones lineales continuas
        predicciones = self.model.predict(X)
        # Truncamos las predicciones en el rango [0.0, 1.0] para respetar el dominio de la LGD
        return np.clip(predicciones, 0.0, 1.0)

def entrenar_modelo_referencia():
    # 1. Leemos el DataFrame de la LGD procesada parseando la columna de fechas
    df = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])

    # 2. Aplicamos la particion temporal (obtenemos un diccionario de DataFrames)
    partes = partir(df)
    train = partes["entrenamiento"]

    # 3. Instanciamos el preprocesador oficial
    prep = Preprocesador()

    # 4. Separamos las variables predictoras (X) y el objetivo (y)
    X_train_raw = train.drop(columns=["target_lgd", "issue_d"])
    y_train = train["target_lgd"].values

    # 5. Ajustamos el preprocesador con X_train y transformamos
    prep.fit(X_train_raw)
    X_train = prep.transform(X_train_raw)

    # 6. Instanciamos y entrenamos la regresion Ridge acotada
    modelo = RegresionRidgeAcotada(alpha=1.0)
    modelo.fit(X_train, y_train)

    # 7. Creamos la carpeta modelos si no existe y guardamos los artefactos
    os.makedirs("modelos", exist_ok=True)
    joblib.dump(modelo, "modelos/regresion_lineal.joblib")
    joblib.dump(prep, "modelos/preprocesador_lgd.joblib")

    print(
        "[OK] Modelo de referencia (Regresion Lineal/Ridge) entrenado y guardado exitosamente."
    )

# Punto de entrada de ejecucion independiente
if __name__ == "__main__":
    # Ejecutamos el entrenamiento del modelo de referencia
    entrenar_modelo_referencia()