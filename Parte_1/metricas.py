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


def tabla_metricas(y, probs):
    """y y probs son diccionarios {particion: arreglo}. Devuelve una tabla con una fila por partición."""
    return pd.DataFrame({k: evaluar(y[k], probs[k]) for k in y}).T