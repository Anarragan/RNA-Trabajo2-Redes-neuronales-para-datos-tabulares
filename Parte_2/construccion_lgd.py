import argparse
import os
import sys

import numpy as np
import pandas as pd

# Aseguramos que el directorio de trabajo sea Parte_2 y añadimos Parte_1 a la ruta de Python
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Parte_1")))


def construir_base_lgd(path_base_lgd="../Parte_1/data/base_lgd.csv", path_salida="resultados/lgd_procesado.csv"):
    """
    Construye la variable objetivo target_lgd a partir de los datos de recuperación post-default.
    Elimina variables posteriores al crédito para evitar fuga de información y guarda resultados/lgd_procesado.csv.

    Args:
        path_base_lgd (str): Ruta al archivo base_lgd.csv generado en Parte 1.
        path_salida (str): Ruta de destino para el dataset procesado de LGD.

    Returns:
        pd.DataFrame: DataFrame procesado con variables predictoras y target_lgd acotada en [0.0, 1.0].
    """
    if not os.path.exists(path_base_lgd):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada '{path_base_lgd}'. "
            "Asegúrese de haber ejecutado previamente la preparación de datos en Parte 1: "
            "'cd ../Parte_1 && python main.py datos'."
        )

    # Cargamos el CSV de creditos en mora de Parte_1 convirtiendo issue_d a fecha
    df = pd.read_csv(path_base_lgd, parse_dates=["issue_d"])

    # Calculamos el recaudo neto post-default: principal + recuperaciones - tarifa agencias
    recaudo_neto = (
        df["total_rec_prncp"]
        + df["recoveries"]
        - df["collection_recovery_fee"]
    )

    # Calculamos la LGD bruta como la fracción no recuperada del monto otorgado
    lgd_bruta = (df["funded_amnt"] - recaudo_neto) / df["funded_amnt"]

    # Truncamos los valores para asegurar que LGD quede estrictamente en el rango [0.0, 1.0]
    df["target_lgd"] = np.clip(lgd_bruta, 0.0, 1.0)

    # Lista de columnas de recaudo posteriores al credito que causarian fuga de informacion
    # funded_amnt se conserva como la exposicion al default (EAD) requerida para el calculo de E[L]
    cols_fuga = [
        "id",
        "loan_status",
        "out_prncp",
        "total_pymnt",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
    ]

    # Eliminamos las columnas posteriores al crédito dejando solo las variables predictoras
    cols_a_eliminar = [c for c in cols_fuga if c in df.columns]
    df_limpio = df.drop(columns=cols_a_eliminar)

    # Verificamos si existe la carpeta de salida y la creamos si no existe
    os.makedirs(os.path.dirname(path_salida), exist_ok=True)

    # Guardamos el dataset procesado con la variable objetivo LGD
    df_limpio.to_csv(path_salida, index=False)

    print(
        f"[OK] Base LGD construida exitosamente con {len(df_limpio):,} registros en '{path_salida}'."
    )
    return df_limpio


def main():
    parser = argparse.ArgumentParser(description="Paso 0 (Parte 2): Construir base LGD a partir de los datos de Parte 1")
    parser.add_argument(
        "--input",
        default="../Parte_1/data/base_lgd.csv",
        help="ruta al archivo base_lgd.csv generado en Parte 1 (defecto: ../Parte_1/data/base_lgd.csv)",
    )
    parser.add_argument(
        "--salida",
        default="resultados/lgd_procesado.csv",
        help="ruta del archivo de salida (defecto: resultados/lgd_procesado.csv)",
    )
    args = parser.parse_args()
    construir_base_lgd(path_base_lgd=args.input, path_salida=args.salida)


if __name__ == "__main__":
    main()
