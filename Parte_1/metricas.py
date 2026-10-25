"""
Métricas de evaluación de la probabilidad de incumplimiento (las mismas para todos los modelos).

    AUC     área bajo la curva ROC (capacidad de ordenar malos por encima de buenos)
    Gini    2*AUC - 1
    KS      máxima distancia entre las distribuciones acumuladas de malos y buenos
    Brier   error cuadrático medio de las probabilidades (mide calibración; menor es mejor)
Además se reporta la tasa observada de malos y la probabilidad media predicha: si la segunda
queda por debajo de la primera, el modelo subestima el riesgo.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def evaluar(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": float(np.max(tpr - fpr)),
            "Brier": brier_score_loss(y, p), "tasa_obs": float(np.mean(y)), "prob_media": float(np.mean(p))}


def curva_calibracion(y, p, n_grupos=10):
    """Curva de calibración: ordena por probabilidad, corta en grupos de igual tamaño (deciles)
    y compara la probabilidad media predicha con la tasa observada de malos de cada grupo.
    Devuelve (prob_media_por_grupo, tasa_observada_por_grupo). Si el modelo está calibrado, son iguales."""
    y, p = np.asarray(y), np.asarray(p)
    grupos = np.array_split(np.argsort(p), n_grupos)
    return np.array([p[g].mean() for g in grupos]), np.array([y[g].mean() for g in grupos])


def tabla_metricas(y, probs):
    """y y probs son diccionarios {particion: arreglo}. Devuelve una tabla con una fila por partición."""
    return pd.DataFrame({k: evaluar(y[k], probs[k]) for k in y}).T