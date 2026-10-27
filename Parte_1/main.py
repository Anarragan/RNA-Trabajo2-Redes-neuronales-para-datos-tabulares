"""
Punto de entrada de la Parte 1: python main.py <comando> [opciones]

Comandos (cada uno corre su script y le pasa el resto de las opciones):
  datos      preparacion_datos.py        loan.csv (Kaggle) -> data/datos_modelo.csv y data/base_lgd.csv
  rl         regresion_logistica.py      modelos/regresion_logistica.joblib, resultados/t3_resumen.csv
  rna        red_neuronal.py             modelos/red_neuronal.pt, resultados/t4_*
  evaluar    evaluacion.py               resultados/t5_*, figuras/t5_*
  scorecard  scorecard_red_neuronal.py   modelos/scorecard.joblib, resultados/t6_rn_puntos_variables.csv
  comparar   comparar.py                 resultados/comparacion.csv
  todo       rl, rna, evaluar, scorecard y comparar, en ese orden (no incluye 'datos')

Ejemplos:
  python main.py todo                          flujo completo (~5 min)
  python main.py todo --demo                   rl, rna y evaluar en versión rápida (< 2 min)
  python main.py rna --demo                    solo la red, rápida
  python main.py rna --busqueda 0 --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4
  python main.py scorecard --base 600 --odds 5 --pdo 20
  python main.py rl --C 0.1
Opciones de un comando:  python main.py <comando> --help
"""
import argparse
import os
import subprocess
import sys
import time

CARPETA = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {
    "datos": "preparacion_datos.py",
    "rl": "regresion_logistica.py",
    "rna": "red_neuronal.py",
    "evaluar": "evaluacion.py",
    "scorecard": "scorecard_red_neuronal.py",
    "comparar": "comparar.py",
}
TODO = ["rl", "rna", "evaluar", "scorecard", "comparar"]
SOLO_DEMO = ["rl", "rna", "evaluar"]   # con 'todo --demo' los demás pasos usarían el modelo real, no el de demostración


def correr(comando, extra):
    print(f"\n=== {comando}: {SCRIPTS[comando]} {' '.join(extra)} ===", flush=True)
    t0 = time.perf_counter()
    if subprocess.run([sys.executable, os.path.join(CARPETA, SCRIPTS[comando]), *extra], cwd=CARPETA).returncode != 0:
        sys.exit(f"El comando '{comando}' falló.")
    print(f"[{comando}: {time.perf_counter() - t0:.0f} s]")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, usage="python main.py <comando> [opciones]")
    p.add_argument("comando", choices=[*SCRIPTS, "todo"], metavar="comando", help="datos | rl | rna | evaluar | scorecard | comparar | todo")
    p.add_argument("opciones", nargs=argparse.REMAINDER, help="se pasan tal cual al script del comando")
    a = p.parse_args()
    if a.comando != "todo":
        return correr(a.comando, a.opciones)
    if set(a.opciones) - {"--demo"}:
        sys.exit("'todo' solo acepta --demo. Para otras opciones corre cada comando por separado.")
    demo = "--demo" in a.opciones
    for c in (SOLO_DEMO if demo else TODO):
        # rl no tiene --demo (es instantáneo); rna y evaluar sí
        correr(c, ["--demo"] if demo and c in ("rna", "evaluar") else [])


if __name__ == "__main__":
    main()
