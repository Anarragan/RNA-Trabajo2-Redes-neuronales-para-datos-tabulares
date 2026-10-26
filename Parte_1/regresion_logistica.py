"""
Tarea 3: regresión logística de referencia. Guarda modelos/regresion_logistica.joblib.

    python regresion_logistica.py
    python regresion_logistica.py --C 0.1
"""
import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from metricas import tabla_metricas
from preprocesamiento import preparar

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # las rutas (data/, modelos/) son relativas a Parte_1

parser = argparse.ArgumentParser(description="Regresión logística de referencia")
parser.add_argument("--C", type=float, default=1.0, help="inverso de la penalización L2 (menor = más penalización)")
args = parser.parse_args()

# 1. Datos: partición temporal + preprocesamiento ajustado solo con entrenamiento ------------
prep, X, y, partes = preparar()
print("PARTICIONES")
for nombre, parte in partes.items():
    print(f"  {nombre:<13} {len(parte):>8,} registros | tasa de malos {parte['malo'].mean():.3f} "
          f"| {parte['issue_d'].min():%Y-%m} a {parte['issue_d'].max():%Y-%m}")

# 2. Ajuste ---------------------------------------------------------------------------------
t0 = time.perf_counter()
modelo = LogisticRegression(C=args.C, max_iter=1000).fit(X["entrenamiento"], y["entrenamiento"])
segundos = time.perf_counter() - t0
n_var = X["entrenamiento"].shape[1]
print(f"\nMODELO: {n_var} variables + intercepto = {n_var + 1} parametros | C = {args.C} | "
      f"1 configuracion | {segundos:.1f} s")

# 3. Evaluación -----------------------------------------------------------------------------
probs = {k: modelo.predict_proba(X[k])[:, 1] for k in X}
print("\nEVALUACION")
print(tabla_metricas(y, probs).round(4).to_string())

# 4. Coeficientes ---------------------------------------------------------------------------
# Continuas y conteos estandarizados: razón de odds por cada +1 desviación estándar.
# Las columnas 0/1 se leen frente a su categoría de referencia.
coef = pd.Series(modelo.coef_[0], index=prep.columnas_)
tabla = pd.DataFrame({"coeficiente": coef, "razon_de_odds": np.exp(coef)})
print("\nCOEFICIENTES (ordenados por tamaño del efecto)")
print(tabla.reindex(coef.abs().sort_values(ascending=False).index).round(3).to_string())

# 5. Guardar el modelo (la scorecard y la aplicación web lo cargan sin reentrenar) ---------------
Path("modelos").mkdir(exist_ok=True)
joblib.dump({"preprocesador": prep, "modelo": modelo}, "modelos/regresion_logistica.joblib")
print("\nModelo guardado en modelos/regresion_logistica.joblib")