# Importamos os para crear carpetas de salida en el sistema de archivos
import os

# Importamos sys para poder modificar las rutas de busqueda de módulos en Python
import sys

# Importamos numpy para operaciones numéricas y el truncamiento de valores (clip)
import numpy as np

# Importamos pandas para la manipulación y procesamiento de las tablas de datos
import pandas as pd

# Añadimos la carpeta Parte_1 a la ruta de Python para reutilizar sus módulos
sys.path.append("../Parte_1")

# Importamos la función de partición temporal unificada desde Parte_1/preprocesamiento.py
from preprocesamiento import partir


def construir_base_lgd(path_base_lgd="../Parte_1/data/base_lgd.csv"):
    # Cargamos el CSV de creditos en mora de Parte_1 convirtiendo issue_d a fecha
    df = pd.read_csv(path_base_lgd, parse_dates=["issue_d"])

    # Calculamos el recaudo neto post-default: principal + recuperaciones - honorarios
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
    cols_fuga = [
        "id",
        "loan_status",
        "funded_amnt",
        "out_prncp",
        "total_pymnt",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
    ]

    # Eliminamos las columnas posteriores al credito dejando solo las variables predictoras
    df_limpio = df.drop(columns=cols_fuga)

    # Verificamos si existe la carpeta resultados y la creamos si no existe
    os.makedirs("resultados", exist_ok=True)

    # Guardamos el dataset procesado con la variable objetivo LGD en resultados/
    df_limpio.to_csv("resultados/lgd_procesado.csv", index=False)

    # Mostramos un mensaje de confirmacion con la cantidad de registros procesados
    print(
        f"[OK] Base LGD construida exitosamente con {len(df_limpio)} registros."
    )

    # Retornamos el DataFrame procesado
    return df_limpio


# Punto de entrada para ejecutar este archivo de manera independiente
if __name__ == "__main__":
    # Ejecutamos la funcion principal
    construir_base_lgd()