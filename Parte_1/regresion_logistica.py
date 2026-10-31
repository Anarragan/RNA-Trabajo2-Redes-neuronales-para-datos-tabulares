"""
Tarea 3: regresión logística de referencia. Guarda modelos/regresion_logistica.joblib y resultados/t3_resumen.csv.

    python regresion_logistica.py            # busca C con tantos valores como configuraciones prueba la red (mismo presupuesto que la red)
    python regresion_logistica.py --C 0.1    # un valor fijo, sin búsqueda
    python regresion_logistica.py --config config.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from metricas import tabla_metricas
from preprocesamiento import preparar
from red_neuronal import ETAPAS

parser = argparse.ArgumentParser(description="Regresión logística de referencia")
parser.add_argument("--C", type=float, default=None, help="inverso de la penalización L2 (menor = más penalización); sin él se busca con el mismo presupuesto que la red")
parser.add_argument("--config", type=str, default=None, help="ruta a archivo JSON con parámetro C")
args = parser.parse_args()

ruta_config = os.path.abspath(args.config) if args.config else None
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # las rutas (data/, modelos/) son relativas a Parte_1

C_param = args.C
if ruta_config:
    if not os.path.exists(ruta_config):
        sys.exit(f"Error: No se encontró el archivo de configuración '{args.config}'")
    with open(ruta_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        if "C" in cfg and C_param is None:
            C_param = cfg["C"]

# 1. Datos: partición temporal + preprocesamiento ajustado solo con entrenamiento ------------
prep, X, y, partes = preparar()
print("PARTICIONES")
for nombre, parte in partes.items():
    print(f"  {nombre:<13} {len(parte):>8,} registros | tasa de malos {parte['malo'].mean():.3f} "
          f"| {parte['issue_d'].min():%Y-%m} a {parte['issue_d'].max():%Y-%m}")

# 2. Búsqueda de C (mismo criterio y presupuesto que la red: misma cantidad de configuraciones, pérdida de validación) -----
N_CONFIGS_RED = 1 + sum(len(op) for _, op in ETAPAS)   # configuraciones que prueba la búsqueda por etapas de la red
valores_C = [C_param] if C_param is not None else list(np.logspace(-4, 1, N_CONFIGS_RED))
filas, ajustes = [], []
for C in valores_C:
    t0 = time.perf_counter()
    m = LogisticRegression(C=C, max_iter=1000).fit(X["entrenamiento"], y["entrenamiento"])
    seg = time.perf_counter() - t0
    perdida = log_loss(y["validacion"], m.predict_proba(X["validacion"])[:, 1])
    filas.append({"C": C, "perdida_val": perdida, "segundos": seg})
    ajustes.append(m)
    print(f"  C = {C:.5f} | pérdida val {perdida:.4f} | {seg:.2f} s")
busqueda = pd.DataFrame(filas)
mejor = int(busqueda["perdida_val"].idxmin())
modelo, C_elegido, segundos = ajustes[mejor], valores_C[mejor], filas[mejor]["segundos"]
n_var = X["entrenamiento"].shape[1]
print(f"\nMODELO: {n_var} variables + intercepto = {n_var + 1} parametros | C elegido = {C_elegido:.5f} | "
      f"{len(valores_C)} configuraciones | búsqueda {busqueda['segundos'].sum():.1f} s")

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

# 6. Búsqueda y resumen para la tabla comparativa (comparar) --------------------------------------
Path("resultados").mkdir(exist_ok=True)
busqueda.to_csv("resultados/t3_busqueda.csv", index=False)
resumen = tabla_metricas(y, probs)[["AUC", "Gini", "KS", "Brier"]]
resumen.insert(0, "particion", resumen.index)
resumen.insert(0, "configuracion", f"C={C_elegido:.5f}")
resumen.insert(2, "n_configuraciones", len(valores_C))
resumen.insert(3, "tiempo_busqueda_s", round(busqueda["segundos"].sum(), 2))
resumen.insert(4, "tiempo_final_s", round(segundos, 2))
resumen.insert(5, "parametros", n_var + 1)
resumen.to_csv("resultados/t3_resumen.csv", index=False)
print("Guardado: resultados/t3_busqueda.csv y t3_resumen.csv")
