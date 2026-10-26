"""
Tarea 6: scorecard construida sobre la regresión logística (modelos/regresion_logistica.joblib).
No reentrena nada: convierte el logit del modelo en puntos.

Escala (se escribe en un solo sentido: más riesgo = MENOS puntos):
    odds_buenos = (1 - p_malo) / p_malo = exp(-logit)
    Puntaje  = Offset + Factor * ln(odds_buenos)
    Factor   = PDO / ln(2)                      (cada +PDO puntos duplica las odds de buenos)
    Offset   = Base - Factor * ln(odds_ref)     (con odds = odds_ref el puntaje es Base)

Como logit = b0 + sum(bj * xj) (xj ya estandarizada o 0/1):
    Puntaje = (Offset - Factor*b0)  +  sum( -Factor * bj * xj )
               puntos base (intercepto)   puntos de cada variable

    python scorecard.py                                  # Base 600, odds 5:1, PDO 20
    python scorecard.py --base 600 --odds 5 --pdo 20     # parámetros a mano
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from metricas import evaluar
from preprocesamiento import BINARIAS, CATEGORICAS, CON_LOG, CONTEOS, CONTINUAS, preparar


def escala(base, odds_ref, pdo):
    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds_ref)
    return factor, offset


def puntaje_desde_prob(p, factor, offset):
    """Fórmula directa: del p_malo al puntaje (sin pasar por la suma de puntos)."""
    return offset + factor * np.log((1 - p) / p)


def puntos_nivel(prep, coef, var, valores, factor):
    """Puntos de una variable continua/conteo en valores crudos (aplica log, recorte y estandarización)."""
    x = pd.Series(np.asarray(valores, dtype=float))
    if var in CON_LOG:
        x = np.log1p(x)
    x = x.clip(prep.lim_inf_[var], prep.lim_sup_[var])
    z = (x - prep.medias_[var]) / prep.desv_[var]
    return -factor * coef[var] * z


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # las rutas son relativas a Parte_1
    p = argparse.ArgumentParser(description="Tarea 6: scorecard sobre la regresión logística")
    p.add_argument("--base", type=float, default=600, help="puntaje base")
    p.add_argument("--odds", type=float, default=5, help="odds de referencia buenos:malos (5 = 5 buenos por cada malo)")
    p.add_argument("--pdo", type=float, default=20, help="puntos para duplicar las odds")
    a = p.parse_args()
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("figuras", exist_ok=True)

    factor, offset = escala(a.base, a.odds, a.pdo)
    lineas = [f"ESCALA: Base = {a.base:g} puntos con odds {a.odds:g}:1 (p_malo = {1 / (1 + a.odds):.3f}) | PDO = {a.pdo:g}",
              f"        Factor = PDO/ln2 = {factor:.4f} | Offset = Base - Factor*ln(odds) = {offset:.4f}"]

    # 1. Modelo y datos (mismo preprocesamiento y partición temporal que todo lo demás) ---------------
    guardado = joblib.load("modelos/regresion_logistica.joblib")
    prep, modelo = guardado["preprocesador"], guardado["modelo"]
    _, X, y, partes = preparar()
    cols = list(prep.columnas_)
    coef = pd.Series(modelo.coef_[0], index=cols)
    b0 = float(modelo.intercept_[0])
    puntos_base = offset - factor * b0

    # 2. Puntos por variable y puntaje de cada cliente ----------------------------------------------------
    puntos_unidad = -factor * coef   # puntos por +1 DE (continuas) o por tener la categoría/indicador (0/1)
    tipo = {c: "continua (+1 DE)" for c in CONTINUAS + CONTEOS}
    tipo.update({c: "indicador 0/1" for c in BINARIAS + ["emp_length_nulo"]})
    for var, ref in CATEGORICAS.items():
        tipo.update({f"{var}_{c}": f"categoría vs {ref}" for c in prep.categorias_[var] if c != ref})
    tabla = pd.DataFrame({"tipo": pd.Series(tipo), "coeficiente": coef, "razon_de_odds": np.exp(coef),
                          "puntos": puntos_unidad}).loc[cols]
    tabla = tabla.reindex(tabla["puntos"].abs().sort_values(ascending=False).index)
    tabla.round(4).to_csv("resultados/t6_puntos_variables.csv", index_label="variable")

    contrib = {k: -factor * X[k][cols].to_numpy() * coef.to_numpy() for k in X}      # n x 34 puntos
    puntaje = {k: puntos_base + contrib[k].sum(axis=1) for k in X}                    # suma de puntos
    prob = {k: modelo.predict_proba(X[k][cols])[:, 1] for k in X}
    directo = {k: puntaje_desde_prob(prob[k], factor, offset) for k in X}             # fórmula directa

    # 3. Verificaciones ------------------------------------------------------------------------------------
    dif = max(np.abs(puntaje[k] - directo[k]).max() for k in X)
    p_ref = 1 / (1 + a.odds)
    p_doble = 1 / (1 + 2 * a.odds)
    lineas += ["", "VERIFICACIONES",
               f"  suma de puntos vs fórmula directa: diferencia máxima {dif:.2e} puntos "
               f"({'OK' if dif < 1e-6 else 'REVISAR'})",
               f"  odds de referencia ({a.odds:g}:1)  -> {puntaje_desde_prob(p_ref, factor, offset):.2f} puntos (debe ser {a.base:g})",
               f"  odds el doble ({2 * a.odds:g}:1)    -> {puntaje_desde_prob(p_doble, factor, offset):.2f} puntos (debe ser {a.base + a.pdo:g})",
               f"  puntos base (intercepto) = Offset - Factor*b0 = {puntos_base:.2f} "
               f"(puntaje con las continuas en su promedio y todas las categorías/indicadores en 0; "
               f"el puntaje medio real en entrenamiento es {puntaje['entrenamiento'].mean():.2f})"]
    medias = {k: puntaje[k].mean() for k in X}

    # 4. Discriminación del puntaje (debe coincidir con la del modelo) ---------------------------------------
    filas = []
    for k in X:
        # menos puntos = más riesgo, por eso se ordena con -puntaje (el Brier no aplica: el puntaje no es probabilidad)
        auc_pun = roc_auc_score(y[k], -puntaje[k])
        fpr, tpr, _ = roc_curve(y[k], -puntaje[k])
        m_mod = evaluar(y[k], prob[k])
        filas.append({"particion": k, "puntaje_medio": puntaje[k].mean(), "puntaje_min": puntaje[k].min(),
                      "puntaje_max": puntaje[k].max(), "AUC_puntaje": auc_pun, "AUC_modelo": m_mod["AUC"],
                      "KS_puntaje": float(np.max(tpr - fpr)), "KS_modelo": m_mod["KS"],
                      "Gini_puntaje": 2 * auc_pun - 1})
    disc = pd.DataFrame(filas)
    disc.to_csv("resultados/t6_discriminacion.csv", index=False)
    lineas += ["", "PUNTAJE POR PARTICIÓN Y DISCRIMINACIÓN", disc.round(4).to_string(index=False)]

    # 5. Bandas de puntaje (cortes fijos con deciles del entrenamiento) y tasa de malos ----------------------
    cortes = np.unique(np.quantile(puntaje["entrenamiento"], np.linspace(0, 1, 11)[1:-1]))
    bandas = []
    for k in X:
        g = np.digitize(puntaje[k], cortes)     # 0 = banda de menor puntaje (más riesgosa)
        for i in range(len(cortes) + 1):
            m = g == i
            bandas.append({"particion": k, "banda": i + 1,
                           "desde": puntaje[k][m].min() if m.any() else np.nan,
                           "hasta": puntaje[k][m].max() if m.any() else np.nan,
                           "n": int(m.sum()), "tasa_malos": y[k][m].mean() if m.any() else np.nan,
                           "prob_media": prob[k][m].mean() if m.any() else np.nan})
    bandas = pd.DataFrame(bandas)
    bandas.to_csv("resultados/t6_bandas.csv", index=False)
    mono = {k: bool(np.all(np.diff(bandas[bandas.particion == k]["tasa_malos"].to_numpy()) <= 1e-9)) for k in X}
    lineas += ["", f"BANDAS DE PUNTAJE (banda 1 = menos puntos = más riesgo; cortes de deciles de entrenamiento: "
                   f"{np.round(cortes, 1).tolist()})",
               bandas[bandas.particion == "prueba"].round(3).to_string(index=False),
               f"  tasa de malos decreciente al subir el puntaje: {mono}"]

    # 6. Tabla de puntos por variable ---------------------------------------------------------------------------
    lineas += ["", f"PUNTOS POR VARIABLE (ordenado por tamaño; continuas: puntos por +1 DE; 0/1: puntos si vale 1; "
                   f"categoría de referencia = 0 puntos)", tabla.round(2).to_string()]

    # 7. Puntos por nivel en valores crudos para las variables continuas ---------------------------------------
    ent = partes["entrenamiento"]
    niveles = []
    for var in CONTINUAS + CONTEOS:
        qs = [0.10, 0.25, 0.50, 0.75, 0.90]
        vals = np.unique(ent[var].dropna().quantile(qs).to_numpy())
        pts = puntos_nivel(prep, coef, var, vals, factor)
        for v, pt in zip(vals, pts):
            niveles.append({"variable": var, "valor": round(float(v), 2), "puntos": round(float(pt), 2)})
    niveles = pd.DataFrame(niveles)
    niveles.to_csv("resultados/t6_puntos_niveles.csv", index=False)
    lineas += ["", "PUNTOS POR NIVEL (percentiles 10/25/50/75/90 del entrenamiento, valores originales; "
                   "0 = promedio del entrenamiento)",
               niveles.pivot_table(index="variable", columns=niveles.groupby("variable").cumcount(),
                                   values=["valor", "puntos"]).round(2).to_string()]

    texto = "\n".join(lineas)
    open("resultados/t6_scorecard.txt", "w", encoding="utf-8").write(texto)
    print(texto)

    # 8. Figura: distribución del puntaje en prueba y tasa de malos por banda ----------------------------------
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for etiqueta, valor in (("buenos", 0), ("malos", 1)):
        ejes[0].hist(puntaje["prueba"][y["prueba"] == valor], bins=40, alpha=0.6, density=True, label=etiqueta)
    ejes[0].set(xlabel="puntaje", ylabel="densidad", title="Puntaje en prueba (buenos vs malos)")
    ejes[0].legend()
    for k in X:
        b = bandas[bandas.particion == k]
        ejes[1].plot(b["banda"], b["tasa_malos"], marker="o", label=k)
    ejes[1].set(xlabel="banda de puntaje (1 = menos puntos)", ylabel="tasa observada de malos",
                title="Tasa de malos por banda de puntaje")
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig("figuras/t6_scorecard.png", dpi=130)
    print("\nArchivos: resultados/t6_*.csv|txt y figuras/t6_scorecard.png")


if __name__ == "__main__":
    main()
