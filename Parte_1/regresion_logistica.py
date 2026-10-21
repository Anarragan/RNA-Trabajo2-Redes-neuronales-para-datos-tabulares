"""
Parte 1, tarea 3: regresión logística (modelo de referencia).
Usa el módulo preprocesamiento.py (debe estar en la misma carpeta).

Ejecutar:  python regresion_logistica.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from preprocesamiento import Preprocesador, cargar_datos, partir

# Cortes de la partición temporal (entrenamiento < validación < prueba)
CORTE_VAL = "2013-01-01"
CORTE_TEST = "2014-01-01"

# 1. Partición por fecha ------------------------------------------------------
train, val, test = partir(cargar_datos(), CORTE_VAL, CORTE_TEST)
print("PARTICIONES")
for nombre, parte in [("entrenamiento", train), ("validacion", val), ("prueba", test)]:
    print(f"  {nombre:<13} {len(parte):>8,} registros | tasa de malos {parte['malo'].mean():.3f} "
          f"| {parte['issue_d'].min():%Y-%m} a {parte['issue_d'].max():%Y-%m}")

# 2. Preprocesamiento: se ajusta SOLO con entrenamiento ---------------------------
prep = Preprocesador().fit(train)
X = {"entrenamiento": prep.transform(train), "validacion": prep.transform(val), "prueba": prep.transform(test)}
y = {"entrenamiento": train["malo"].to_numpy(), "validacion": val["malo"].to_numpy(), "prueba": test["malo"].to_numpy()}

# 3. Ajuste: regresión logística con penalización L2 (C=1, 1 sola configuración) -----
t0 = time.perf_counter()
modelo = LogisticRegression(C=1.0, max_iter=1000).fit(X["entrenamiento"], y["entrenamiento"])
segundos = time.perf_counter() - t0
print(f"\nMODELO: {X['entrenamiento'].shape[1]} variables + intercepto = "
      f"{X['entrenamiento'].shape[1] + 1} parametros | 1 configuracion | {segundos:.1f} s")

# 4. Evaluación -------------------------------------------------------------------
filas = {}
for nombre in X:
    p = modelo.predict_proba(X[nombre])[:, 1]
    auc = roc_auc_score(y[nombre], p)
    fpr, tpr, _ = roc_curve(y[nombre], p)
    filas[nombre] = {"AUC": auc, "Gini": 2 * auc - 1, "KS": np.max(tpr - fpr),
                     "Brier": brier_score_loss(y[nombre], p),
                     "tasa_obs": y[nombre].mean(), "prob_media": p.mean()}
print("\nEVALUACION")
print(pd.DataFrame(filas).T.round(4).to_string())

# 5. Coeficientes -------------------------------------------------------------------
# Continuas y conteos estandarizados: razón de odds por cada +1 desviación estándar.
coef = pd.Series(modelo.coef_[0], index=prep.columnas_)
tabla = pd.DataFrame({"coeficiente": coef, "razon_de_odds": np.exp(coef)})
tabla = tabla.reindex(coef.abs().sort_values(ascending=False).index)
print("\nCOEFICIENTES (ordenados por tamaño del efecto)")
print(tabla.round(3).to_string())