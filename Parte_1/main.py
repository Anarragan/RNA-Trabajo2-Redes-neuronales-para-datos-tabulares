"""
Corre toda la Parte 1 en orden. Cada paso también se puede correr solo.

    python main.py              # t2, t3, t4, t5 (tarda ~5 min)
    python main.py --demo       # igual pero rápido (< 2 min); no pisa los resultados reales
    python main.py --solo t4    # un solo paso: datos | t2 | t3 | t4 | t5
    python main.py --con-datos  # además rehace los datos desde loan.csv (descarga de Kaggle)
"""
import argparse
import os
import subprocess
import sys
import time

CARPETA = os.path.dirname(os.path.abspath(__file__))
PASOS = {
    "datos": "preparacion_datos.py",      # loan.csv -> data/datos_modelo.csv
    "t2": "analisis_descriptivo.py",      # hipótesis H1-H6
    "t3": "regresion_logistica.py",       # modelo de referencia
    "t4": "red_neuronal.py",              # red neuronal (búsqueda + 30 semillas)
    "t5": "evaluacion.py",                # AUC, Gini, KS, Brier, calibración
}

parser = argparse.ArgumentParser(description="Flujo completo de la Parte 1")
parser.add_argument("--solo", choices=list(PASOS), help="correr un solo paso")
parser.add_argument("--demo", action="store_true", help="versión rápida de t4 y t5")
parser.add_argument("--con-datos", action="store_true", help="incluir el paso 'datos'")
args = parser.parse_args()

pasos = [args.solo] if args.solo else (["datos"] if args.con_datos else []) + ["t2", "t3", "t4", "t5"]
for paso in pasos:
    comando = [sys.executable, os.path.join(CARPETA, PASOS[paso])]
    if args.demo and paso in ("t4", "t5"):
        comando.append("--demo")
    print(f"\n=== {paso}: {PASOS[paso]} ===", flush=True)
    t0 = time.perf_counter()
    if subprocess.run(comando, cwd=CARPETA).returncode != 0:
        sys.exit(f"El paso {paso} falló.")
    print(f"[{paso}: {time.perf_counter() - t0:.0f} s]")
