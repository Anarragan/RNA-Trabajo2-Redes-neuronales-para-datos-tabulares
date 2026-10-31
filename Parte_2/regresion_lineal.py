import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

# Aseguramos que el directorio de trabajo sea Parte_2 y añadimos Parte_1 a la ruta para reutilizar sus modulos
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Parte_1")))

# Importamos la particion temporal y el preprocesador oficial de la Parte 1
from preprocesamiento import Preprocesador, partir


class RegresionRidgeAcotada(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        predicciones = self.model.predict(X)
        # Truncamos las predicciones en el rango [0.0, 1.0] para respetar el dominio de la LGD
        return np.clip(predicciones, 0.0, 1.0)


def cargar_modelo_referencia(ruta="modelos/regresion_lineal.joblib"):
    """Carga el modelo de referencia garantizando compatibilidad de serialización."""
    import __main__
    __main__.RegresionRidgeAcotada = RegresionRidgeAcotada
    return joblib.load(ruta)


def entrenar_modelo_referencia(alpha=1.0):
    """
    Entrena el modelo lineal de referencia (Ridge acotado en [0, 1]) para LGD.

    Args:
        alpha (float): Parámetro de penalización L2 para la regresión Ridge (defecto: 1.0).

    Returns:
        None: Guarda el modelo entrenado en modelos/regresion_lineal.joblib y el preprocesador en modelos/preprocesador_lgd.joblib.
    """
    df = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])
    partes = partir(df)
    train = partes["entrenamiento"]

    prep = Preprocesador()
    X_train_raw = train.drop(columns=["target_lgd", "issue_d", "funded_amnt"])
    y_train = train["target_lgd"].values

    prep.fit(X_train_raw)
    X_train = prep.transform(X_train_raw)

    # Entrenamiento
    modelo = RegresionRidgeAcotada(alpha=alpha)
    modelo.fit(X_train, y_train)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(modelo, "modelos/regresion_lineal.joblib")
    joblib.dump(prep, "modelos/preprocesador_lgd.joblib")

    n_params = len(modelo.model.coef_) + 1
    print(
        f"[OK] Modelo de referencia (Regresion Lineal/Ridge acotada) entrenado exitosamente. "
        f"Parámetros: {n_params} (alpha={alpha}). Guardado en modelos/regresion_lineal.joblib"
    )


def main():
    parser = argparse.ArgumentParser(description="Tarea 2 (Parte 2): Modelo de referencia Ridge para LGD")
    parser.add_argument("--alpha", type=float, default=1.0, help="inverso de la regularización L2 / alpha de Ridge (defecto: 1.0)")
    parser.add_argument("--config", type=str, default=None, help="ruta a archivo JSON con parámetro alpha")
    args = parser.parse_args()

    ruta_config = os.path.abspath(args.config) if args.config else None
    alpha = args.alpha
    if ruta_config:
        if not os.path.exists(ruta_config):
            sys.exit(f"Error: No se encontró el archivo de configuración '{args.config}'")
        with open(ruta_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if "alpha" in cfg:
                alpha = cfg["alpha"]

    entrenar_modelo_referencia(alpha=alpha)


if __name__ == "__main__":
    main()
