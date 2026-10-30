"""
Punto de entrada de la Parte 2: python main.py <comando> [opciones]

Comandos (cada uno corre su script y le pasa el resto de las opciones):
  datos      construccion_lgd.py         data/base_lgd.csv -> resultados/lgd_procesado.csv
  rl         regresion_lineal.py         modelos/regresion_lineal.joblib, preprocesador_lgd.joblib
  rna        red_neuronal_lgd.py         modelos/red_neuronal_lgd.pt, resultados/t4_*
  evaluar    evaluacion.py               figuras/t4_distribucion_lgd.png, resultados/t5_perdida_esperada.csv
  comparar   comparar.py                 resultados/comparacion_lgd.csv
  todo       datos, rl, rna, evaluar y comparar, en ese orden

Ejemplos:
  python main.py todo                          flujo completo con búsqueda e hiperparámetros
  python main.py todo --demo                   versión rápida (< 30 s) para sustentación en vivo
  python main.py rna --demo                    solo la red neuronal de LGD en versión rápida
  python main.py rna --semillas 30 --epocas 40
  python main.py evaluar --demo
  python main.py rl --alpha 1.0
Opciones de un comando:  python main.py <comando> --help
"""
import argparse
import os
import subprocess
import sys
import time

CARPETA = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {
    "datos": "construccion_lgd.py",
    "rl": "regresion_lineal.py",
    "rna": "red_neuronal_lgd.py",
    "evaluar": "evaluacion.py",
    "comparar": "comparar.py",
}
TODO = ["datos", "rl", "rna", "evaluar", "comparar"]
SOLO_DEMO = ["datos", "rl", "rna", "evaluar"]


def correr(comando, extra):
    """
    Ejecuta scripts individualmente de la Parte 2.

    Args:
        comando (str): Clave del comando registrado en SCRIPTS.
        extra (list): Argumentos adicionales para pasarle al script.
    """
    print(f"\n=== {comando}: {SCRIPTS[comando]} {' '.join(extra)} ===", flush=True)
    t0 = time.perf_counter()
    if subprocess.run([sys.executable, os.path.join(CARPETA, SCRIPTS[comando]), *extra], cwd=CARPETA).returncode != 0:
        sys.exit(f"El comando '{comando}' falló.")
    print(f"[{comando}: {time.perf_counter() - t0:.1f} s]")


def main():
    """Punto de entrada principal para despachar comandos en la Parte 2."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python main.py <comando> [opciones]",
    )
    p.add_argument(
        "comando",
        choices=[*SCRIPTS, "todo"],
        metavar="comando",
        help="datos | rl | rna | evaluar | comparar | todo",
    )
    p.add_argument("opciones", nargs=argparse.REMAINDER, help="se pasan tal cual al script del comando")
    a = p.parse_args()
    if a.comando != "todo":
        return correr(a.comando, a.opciones)
    if set(a.opciones) - {"--demo"}:
        sys.exit("'todo' solo acepta --demo. Para otras opciones corre cada comando por separado.")
    demo = "--demo" in a.opciones
    for c in (SOLO_DEMO if demo else TODO):
        correr(c, ["--demo"] if demo and c in ("rna", "evaluar") else [])


if __name__ == "__main__":
    main()
