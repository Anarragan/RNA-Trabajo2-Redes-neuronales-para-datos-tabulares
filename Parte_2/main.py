# Importamos argparse para la gestion de argumentos desde la linea de comandos
import argparse

# Importamos os para verificaciones de archivos y carpetas
import os

# Importamos sys para manejar las rutas y salidas del sistema
import sys

# Importamos time para medir el tiempo total de ejecucion del pipeline
import time

# Importamos la funcion de construccion de la target LGD del archivo 1
from construccion_lgd import construir_base_lgd

# Importamos la funcion de evaluacion final y calculo de E[L] del archivo 4
from evaluacion_lgd import evaluar_modelos_y_calcular_el

# Importamos la funcion de entrenamiento sobre N semillas del archivo 3
from red_neuronal_lgd import entrenar_30_semillas

# Importamos la funcion de entrenamiento del modelo de referencia del archivo 2
from regresion_lineal import entrenar_modelo_referencia


def main():
    # Creamos el parser de argumentos de linea de comandos
    parser = argparse.ArgumentParser(
        description="Flujo principal de la Parte 2: LGD y Perdida Esperada E[L]"
    )

    # Añadimos la bandera --demo para ejecuciones de prueba rapida (< 2 minutos)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecuta una version rapida con 3 semillas para demostracion en vivo",
    )

    # Procesamos los argumentos ingresados por el usuario
    args = parser.parse_args()

    # Registramos el tiempo inicial de ejecucion
    tiempo_inicio = time.time()

    # Imprimimos encabezado del pipeline
    print("=" * 70)
    print("  EJECUCIÓN PARTE 2: PÉRDIDA DADO EL INCUMPLIMIENTO (LGD) Y E[L]")
    print("=" * 70)

    # Paso 1: Construccion de la base y calculo de target_lgd
    print("\n[Paso 1/4] Construyendo variable objetivo LGD y filtrando datos...")
    construir_base_lgd()

    # Paso 2: Entrenamiento del modelo de referencia
    print("\n[Paso 2/4] Entrenando modelo de referencia (Regresion Lineal/Ridge)...")
    entrenar_modelo_referencia()

    # Determinamos la cantidad de semillas segun la presencia de la bandera --demo
    num_semillas = 3 if args.demo else 30

    # Paso 3: Entrenamiento de la Red Neuronal sobre N semillas
    print(
        f"\n[Paso 3/4] Entrenando Red Neuronal MLP sobre {num_semillas} semillas..."
    )
    entrenar_30_semillas(num_semillas=num_semillas)

    # Paso 4: Evaluacion final, graficos de densidad y Perdida Esperada
    print("\n[Paso 4/4] Generando evaluacion final, gráficos y Pérdida Esperada...")
    evaluar_modelos_y_calcular_el()

    # Calculamos la duracion total del proceso en segundos
    duracion_total = time.time() - tiempo_inicio

    # Imprimimos mensaje final de exito con el tiempo total transcurrido
    print("\n" + "=" * 70)
    print(
        f"  [ÉXITO] PARTE 2 COMPLETADA SATISFACTORIAMENTE EN {duracion_total:.2f} SEGUNDOS"
    )
    print("=" * 70)


# Punto de entrada principal del script orquestador
if __name__ == "__main__":
    # Ejecutamos la funcion main
    main()