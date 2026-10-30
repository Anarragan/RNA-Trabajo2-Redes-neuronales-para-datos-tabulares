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

import numpy as np

def matriz_confusion(y, p, t):
    pred = (p >= t).astype(int)          # función indicadora: 1[p >= t]
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    n = len(y)
    return {"umbral": t, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "exactitud": (tp + tn) / n,
            "sensibilidad": tp / (tp + fn) if tp + fn else np.nan,
            "especificidad": tn / (tn + fp) if tn + fp else np.nan,
            "pct_rechazo": (tp + fp) / n,
            "mora_en_aprobados": fn / (fn + tn) if fn + tn else np.nan}

def umbral_por_costos(y_val, p_val, razon):
    """Minimiza FN*razon + FP en validación. Grilla por cuantiles: sirve aunque las p sean pequeñas."""
    grilla = np.unique(np.quantile(p_val, np.linspace(0.01, 0.99, 197)))
    costos = [(lambda m: razon * m["FN"] + m["FP"])(matriz_confusion(y_val, p_val, t)) for t in grilla]
    return grilla[int(np.argmin(costos))]


def graficar_confusion(conf, ruta, particion="prueba"):
    """Una matriz de confusión por modelo (fila) y razón de costos (columna). Cada celda muestra el
    conteo y el porcentaje de su fila (de los malos reales, cuántos se detectan; de los buenos, cuántos se aprueban)."""
    conf = conf[conf["particion"] == particion]
    modelos, razones = list(conf["modelo"].unique()), sorted(conf["razon_costos"].unique())
    fig, ejes = plt.subplots(len(modelos), len(razones), figsize=(3.6 * len(razones), 3.4 * len(modelos)), squeeze=False)
    for i, modelo in enumerate(modelos):
        for j, razon in enumerate(razones):
            r = conf[(conf["modelo"] == modelo) & (conf["razon_costos"] == razon)].iloc[0]
            m = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])   # filas: real bueno/malo; columnas: aprobar/rechazar
            pct = m / m.sum(axis=1, keepdims=True)
            eje = ejes[i, j]
            eje.imshow(pct, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    eje.text(b, a, f"{m[a, b]:,}\n{pct[a, b]:.1%}", ha="center", va="center",
                             color="white" if pct[a, b] > 0.5 else "black")
            eje.set_xticks([0, 1], ["aprobar", "rechazar"])
            eje.set_yticks([0, 1], ["bueno", "malo"])
            eje.set_title(f"{modelo}\nFN = {razon}x FP | umbral {r['umbral']:.3f}", fontsize=9)
            if j == 0:
                eje.set_ylabel("real")
            if i == len(modelos) - 1:
                eje.set_xlabel("decisión")
    fig.suptitle(f"Matriz de confusión en {particion} (umbral elegido en validación; % por fila real)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


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
    
    # --- dentro de main(), después de las métricas ---
    filas = []
    for modelo, p_modelo in probs.items():
        for razon in [2, 3, 5, 10]:
            t = umbral_por_costos(y["validacion"], p_modelo["validacion"], razon)
            for nombre in ["validacion", "prueba"]:
                filas.append({"modelo": modelo, "razon_costos": razon, "particion": nombre,
                            **matriz_confusion(y[nombre], p_modelo[nombre], t)})
    conf = pd.DataFrame(filas)
    conf.to_csv(f"resultados/t5_confusion{sufijo}.csv", index=False)
    print("\nMATRIZ DE CONFUSIÓN (umbral elegido en validación)\n" + conf.round(3).to_string(index=False))

    graficar_confusion(conf, f"figuras/t5_confusion{sufijo}.png")

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
    print(f"\nFiguras en figuras/t5_confusion{sufijo}.png, figuras/t5_calibracion{sufijo}.png y figuras/t5_roc{sufijo}.png")


if __name__ == "__main__":
    main()
