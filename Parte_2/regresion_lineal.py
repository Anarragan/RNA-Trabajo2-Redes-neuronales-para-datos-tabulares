import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Aseguramos que el directorio de trabajo sea Parte_2 y añadimos Parte_1 a la ruta para reutilizar sus modulos
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Parte_1")))

# Importamos la particion temporal y el preprocesador oficial de la Parte 1
from preprocesamiento import Preprocesador, partir
from red_neuronal_lgd import ETAPAS


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


def entrenar_modelo_referencia(alpha=None):
    """
    Entrena el modelo lineal de referencia (Ridge acotado en [0, 1]) para LGD.

    Args:
        alpha (float | None): penalización L2 de Ridge. Si es None, se busca entre tantos valores como configuraciones prueba la red, por MSE de validación
            (mismo presupuesto y criterio que la red neuronal).

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

    # Búsqueda de alpha (mismo número de configuraciones que la red, criterio: MSE de validación, igual que la red)
    val = partes["validacion"]
    X_val = prep.transform(val.drop(columns=["target_lgd", "issue_d", "funded_amnt"]))
    y_val = val["target_lgd"].values
    valores_alpha = [alpha] if alpha is not None else list(np.logspace(-2, 4, 1 + sum(len(op) for _, op in ETAPAS)))   # mismo número de configuraciones que la red
    filas, ajustes = [], []
    for a in valores_alpha:
        t0 = time.perf_counter()
        m = RegresionRidgeAcotada(alpha=a).fit(X_train, y_train)
        seg = time.perf_counter() - t0
        filas.append({"alpha": a, "loss_mse_val": mean_squared_error(y_val, m.predict(X_val)), "segundos": seg})
        ajustes.append(m)
    busqueda = pd.DataFrame(filas)
    mejor = int(busqueda["loss_mse_val"].idxmin())
    modelo, alpha, seg_final = ajustes[mejor], valores_alpha[mejor], filas[mejor]["segundos"]

    os.makedirs("resultados", exist_ok=True)
    busqueda.to_csv("resultados/t2_busqueda_ridge.csv", index=False)
    pd.DataFrame([{"n_configuraciones": len(valores_alpha), "alpha": alpha,
                   "tiempo_busqueda_s": round(busqueda["segundos"].sum(), 4),
                   "tiempo_final_s": round(seg_final, 4)}]).to_csv("resultados/t2_resumen_ridge.csv", index=False)

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
    parser.add_argument("--alpha", type=float, default=None, help="alpha de Ridge (penalización L2). Sin este valor se busca con el mismo presupuesto que la red")
    parser.add_argument("--config", type=str, default=None, help="ruta a archivo JSON con parámetro alpha")
    args = parser.parse_args()

    alpha = args.alpha
    if args.config:
        if not os.path.exists(args.config):
            sys.exit(f"Error: No se encontró el archivo de configuración '{args.config}'")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if "alpha" in cfg and alpha is None:
                alpha = cfg["alpha"]

    entrenar_modelo_referencia(alpha=alpha)


if __name__ == "__main__":
    main()
