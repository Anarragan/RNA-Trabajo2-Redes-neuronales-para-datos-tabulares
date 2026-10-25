"""
Tarea 5: compara regresión logística y red (AUC, Gini, KS, Brier, curva de calibración, ROC)
con la partición temporal. No reentrena: carga los modelos de modelos/.

    python evaluacion.py
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # guarda las figuras en archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from metricas import curva_calibracion, tabla_metricas
from preprocesamiento import preparar
from red_neuronal import cargar_red, contar_parametros, predecir


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # las rutas son relativas a Parte_1
    p = argparse.ArgumentParser(description="Tarea 5: evaluación de los modelos")
    p.add_argument("--demo", action="store_true", help="usar el modelo de red guardado en modo demo")
    a = p.parse_args()
    sufijo = "_demo" if a.demo else ""
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("figuras", exist_ok=True)

    # 1. Datos y modelos ya entrenados (no se reentrena nada) ------------------------------------------
    prep, X, y, partes = preparar()
    reg_log = joblib.load("modelos/regresion_logistica.joblib")["modelo"]
    red, config = cargar_red(f"modelos/red_neuronal{sufijo}.pt")
    probs = {
        "Regresión logística": {k: reg_log.predict_proba(X[k])[:, 1] for k in X},
        "Red neuronal": {k: predecir(red, X[k]) for k in X},
    }

    # 2. Particiones (justificación de la partición temporal) ---------------------------------------
    particiones = pd.DataFrame({
        k: {"registros": len(v), "tasa_malos": v["malo"].mean(),
            "desde": f"{v['issue_d'].min():%Y-%m}", "hasta": f"{v['issue_d'].max():%Y-%m}"}
        for k, v in partes.items()}).T
    particiones.to_csv("resultados/t5_particiones.csv")
    print("PARTICIONES (por fecha de originación)\n" + particiones.to_string())

    # 3. Métricas de discriminación (AUC, Gini, KS) y calibración (Brier) ------------------------------
    tablas = []
    for modelo, p_modelo in probs.items():
        t = tabla_metricas(y, p_modelo)
        t.insert(0, "modelo", modelo)
        t.insert(1, "particion", t.index)
        tablas.append(t)
    metricas = pd.concat(tablas, ignore_index=True)
    metricas.to_csv(f"resultados/t5_metricas{sufijo}.csv", index=False)
    print("\nMÉTRICAS\n" + metricas.round(4).to_string(index=False))
    print(f"\nParámetros entrenables: regresión logística {len(reg_log.coef_[0]) + 1} | "
          f"red neuronal {contar_parametros(red)} ({config['capas']})")

    # 4. Curva de calibración: probabilidad predicha vs. tasa observada, por deciles ------------------
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for eje, nombre in zip(ejes, X):
        for modelo, p_modelo in probs.items():
            predicha, observada = curva_calibracion(y[nombre], p_modelo[nombre])
            eje.plot(predicha, observada, marker="o", label=modelo)
        eje.plot([0, 0.6], [0, 0.6], "k--", label="calibración perfecta")
        eje.set_title(nombre)
        eje.set_xlabel("probabilidad predicha (media del decil)")
    ejes[0].set_ylabel("tasa observada de malos")
    ejes[0].legend()
    fig.suptitle("Curva de calibración por partición temporal")
    fig.tight_layout()
    fig.savefig(f"figuras/t5_calibracion{sufijo}.png", dpi=130)

    # 5. Curvas ROC en prueba -----------------------------------------------------------------------
    fig, eje = plt.subplots(figsize=(5, 5))
    for modelo, p_modelo in probs.items():
        fpr, tpr, _ = roc_curve(y["prueba"], p_modelo["prueba"])
        eje.plot(fpr, tpr, label=modelo)
    eje.plot([0, 1], [0, 1], "k--")
    eje.set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos", title="ROC en prueba")
    eje.legend()
    fig.tight_layout()
    fig.savefig(f"figuras/t5_roc{sufijo}.png", dpi=130)
    print(f"\nFiguras en figuras/t5_calibracion{sufijo}.png y figuras/t5_roc{sufijo}.png")


if __name__ == "__main__":
    main()
