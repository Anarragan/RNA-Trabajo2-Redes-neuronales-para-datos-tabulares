import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

from red_neuronal_lgd import cargar_red_lgd, contar_parametros
from regresion_lineal import cargar_modelo_referencia
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Parte_1")))
from preprocesamiento import partir


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("resultados", exist_ok=True)

    # 1. Cargar datos
    df = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])
    partes = partir(df)
    prep = joblib.load("modelos/preprocesador_lgd.joblib")

    X = {k: prep.transform(partes[k].drop(columns=["target_lgd", "issue_d", "funded_amnt"])) for k in partes}
    y = {k: partes[k]["target_lgd"].values for k in partes}

    # 2. Modelo de referencia Ridge
    modelo_base = cargar_modelo_referencia("modelos/regresion_lineal.joblib")
    n_params_base = len(modelo_base.model.coef_) + 1

    filas = []
    for part in ["entrenamiento", "validacion", "prueba"]:
        preds = modelo_base.predict(X[part])
        mae = mean_absolute_error(y[part], preds)
        rmse = root_mean_squared_error(y[part], preds)
        filas.append({
            "modelo": "Regresión Ridge",
            "particion": part,
            "n_configuraciones": 1,
            "parametros": n_params_base,
            "n_semillas": 1,
            "MAE_media": mae,
            "MAE_desv": "n/a",
            "MAE_mejor": mae,
            "MAE_peor": mae,
            "RMSE_media": rmse,
            "RMSE_desv": "n/a",
            "RMSE_mejor": rmse,
            "RMSE_peor": rmse,
        })

    # 3. Red Neuronal MLP
    modelo_nn, config_nn = cargar_red_lgd("modelos/red_neuronal_lgd.pt")
    n_params_nn = contar_parametros(modelo_nn)

    busqueda = pd.read_csv("resultados/t4_busqueda_lgd.csv") if os.path.exists("resultados/t4_busqueda_lgd.csv") else pd.DataFrame([{"segundos": 0}])
    semillas = pd.read_csv("resultados/t4_semillas_lgd.csv")

    for part, col_mae, col_rmse in [("validacion", "MAE_val", "RMSE_val"), ("prueba", "MAE_test", "RMSE_test")]:
        filas.append({
            "modelo": f"Red Neuronal ({config_nn['capas']})",
            "particion": part,
            "n_configuraciones": len(busqueda),
            "parametros": n_params_nn,
            "n_semillas": len(semillas),
            "MAE_media": semillas[col_mae].mean(),
            "MAE_desv": semillas[col_mae].std(),
            "MAE_mejor": semillas[col_mae].min(),
            "MAE_peor": semillas[col_mae].max(),
            "RMSE_media": semillas[col_rmse].mean(),
            "RMSE_desv": semillas[col_rmse].std(),
            "RMSE_mejor": semillas[col_rmse].min(),
            "RMSE_peor": semillas[col_rmse].max(),
        })

    df_comp = pd.DataFrame(filas)
    df_comp.to_csv("resultados/comparacion_lgd.csv", index=False)
    print("\n--- COMPARACIÓN MODELOS LGD (Partición de Prueba) ---")
    prueba_df = df_comp[df_comp["particion"] == "prueba"]
    print(prueba_df[["modelo", "parametros", "n_semillas", "MAE_media", "MAE_desv", "RMSE_media", "RMSE_desv"]].to_string(index=False))
    print(f"\n[OK] Tabla comparativa guardada en 'resultados/comparacion_lgd.csv'")


if __name__ == "__main__":
    main()

