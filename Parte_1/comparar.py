"""
Tabla comparativa regresión logística vs red neuronal, solo con salidas ya generadas (no reentrena).
Lee resultados/t3_resumen.csv, t4_busqueda.csv, t4_semillas.csv y modelos/red_neuronal.pt.
Escribe resultados/comparacion.csv: una fila por modelo y partición, con configuraciones, tiempo,
parámetros, semillas y media, desviación, mejor y peor de AUC, Gini, KS y Brier.

    python comparar.py
"""
import os

import pandas as pd

from red_neuronal import cargar_red, contar_parametros

METRICAS = ["AUC", "Gini", "KS", "Brier"]
MEJOR = {"AUC": "max", "Gini": "max", "KS": "max", "Brier": "min"}   # Brier: menor es mejor


def estadisticas(valores, metrica):
    """media, desviación, mejor y peor de una métrica (la desviación es 'n/a' con un solo valor)."""
    mejor, peor = (valores.max(), valores.min()) if MEJOR[metrica] == "max" else (valores.min(), valores.max())
    desv = valores.std() if len(valores) > 1 else "n/a"
    return {f"{metrica}_media": valores.mean(), f"{metrica}_desv": desv,
            f"{metrica}_mejor": mejor, f"{metrica}_peor": peor}


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rl = pd.read_csv("resultados/t3_resumen.csv")
    busqueda = pd.read_csv("resultados/t4_busqueda.csv")
    semillas = pd.read_csv("resultados/t4_semillas.csv")
    red, _ = cargar_red("modelos/red_neuronal.pt")
    por_semilla = semillas.drop_duplicates("semilla")   # el tiempo se repite en cada partición
    filas = []
    for particion in rl["particion"]:   # regresión logística: determinista, un solo valor
        r = rl[rl["particion"] == particion].iloc[0]
        filas.append({"modelo": "Regresión logística", "particion": particion,
                      "n_configuraciones": int(r["n_configuraciones"]), "tiempo_busqueda_s": r["tiempo_busqueda_s"],
                      "tiempo_final_s": r["tiempo_final_s"], "tiempo_total_s": round(r["tiempo_busqueda_s"] + r["tiempo_final_s"], 2),
                      "parametros": int(r["parametros"]), "n_semillas": 1,
                      **{k: v for m in METRICAS for k, v in estadisticas(pd.Series([r[m]]), m).items()}})
    t_busqueda = busqueda["segundos"].sum()
    for particion, tabla in semillas.groupby("particion", sort=False):
        filas.append({"modelo": "Red neuronal", "particion": particion,
                      "n_configuraciones": len(busqueda), "tiempo_busqueda_s": round(t_busqueda, 2),
                      "tiempo_final_s": round(por_semilla["segundos"].mean(), 2),   # un entrenamiento (media por semilla)
                      "tiempo_total_s": round(t_busqueda + por_semilla["segundos"].sum(), 2),
                      "parametros": contar_parametros(red), "n_semillas": tabla["semilla"].nunique(),
                      **{k: v for m in METRICAS for k, v in estadisticas(tabla[m], m).items()}})
    out = pd.DataFrame(filas)
    out.to_csv("resultados/comparacion.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out["particion"] == "prueba"].round(4).T.to_string(header=False))
    print(f"\nCondiciones: mismas particiones y preprocesamiento; {rl['n_configuraciones'].iloc[0]} configuraciones (regresión logística) y {len(busqueda)} (red), "
          "elegidas por pérdida de validación.\nLa regresión logística es determinista (sin semillas): su desviación es n/a. "
          "tiempo_total = búsqueda + entrenamientos finales (RNA: 30 semillas).\nTabla completa en resultados/comparacion.csv")


if __name__ == "__main__":
    main()
