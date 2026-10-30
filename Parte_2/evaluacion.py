import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Guarda figuras sin requerir entorno gráfico interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch

# Aseguramos que el directorio de trabajo sea Parte_2 y añadimos Parte_1 a la ruta de módulos
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Parte_1")))

# Importamos la partición temporal unificada
from preprocesamiento import partir

# Importamos utilidades de los modelos de Parte 2
from red_neuronal_lgd import cargar_red_lgd, predecir_lgd, contar_parametros
from regresion_lineal import cargar_modelo_referencia

# Importamos la red de la Parte 1 para obtener la Probabilidad de Incumplimiento (PD)
from red_neuronal import cargar_red as cargar_red_pd, predecir as predecir_pd


def evaluar_modelos_y_calcular_el(demo=False):
    """
    Evalúa los modelos de LGD (Ridge vs Red Neuronal) sobre la partición de prueba,
    genera y actualiza el gráfico de distribuciones, guarda las predicciones
    y calcula la Pérdida Esperada E[L] = PD * LGD * EAD.
    """
    sufijo = "_demo" if demo else ""
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("figuras", exist_ok=True)

    # 1. Cargar datos procesados de LGD y extraer la partición de prueba
    if not os.path.exists("resultados/lgd_procesado.csv"):
        raise FileNotFoundError(
            "No se encontró 'resultados/lgd_procesado.csv'. "
            "Ejecute primero 'python main.py datos'."
        )

    df_lgd = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])
    partes = partir(df_lgd)
    test = partes["prueba"]

    # 2. Cargar el preprocesador oficial y transformar prueba
    prep_lgd = joblib.load("modelos/preprocesador_lgd.joblib")
    X_test_raw = test.drop(columns=["target_lgd", "issue_d", "funded_amnt"])
    y_test = test["target_lgd"].values
    X_test = prep_lgd.transform(X_test_raw)

    # 3. Predicciones del modelo de referencia (Regresión Ridge acotada)
    modelo_base = cargar_modelo_referencia("modelos/regresion_lineal.joblib")
    preds_base = modelo_base.predict(X_test)
    n_params_base = len(modelo_base.model.coef_) + 1

    # 4. Predicciones de la Red Neuronal (MLP para LGD)
    ruta_nn = f"modelos/red_neuronal_lgd{sufijo}.pt"
    if not os.path.exists(ruta_nn):
        ruta_nn = "modelos/red_neuronal_lgd.pt"

    modelo_nn, config_nn = cargar_red_lgd(ruta_nn)
    preds_nn = predecir_lgd(modelo_nn, X_test)
    n_params_nn = contar_parametros(modelo_nn)

    # 5. Métricas de desempeño en prueba (MAE y RMSE)
    mae_base = mean_absolute_error(y_test, preds_base)
    rmse_base = root_mean_squared_error(y_test, preds_base)
    mae_nn = mean_absolute_error(y_test, preds_nn)
    rmse_nn = root_mean_squared_error(y_test, preds_nn)

    print("\n" + "=" * 65)
    print("  EVALUACIÓN DE MODELOS LGD EN PARTICIÓN DE PRUEBA")
    print("=" * 65)
    tabla_resumen = pd.DataFrame({
        "Modelo": ["Regresión Ridge", f"Red Neuronal ({config_nn['capas']})"],
        "Parámetros": [n_params_base, n_params_nn],
        "MAE": [mae_base, mae_nn],
        "RMSE": [rmse_base, rmse_nn],
    })
    print(tabla_resumen.to_string(index=False))

    # 6. Estadísticas descriptivas de las distribuciones
    print("\n--- DISTRIBUCIÓN DE LGD (Valores reales vs Predicciones) ---")
    stats_df = pd.DataFrame({
        "Variable": ["LGD Real Observada", "Predicción Ridge", "Predicción Red Neuronal"],
        "Media": [y_test.mean(), preds_base.mean(), preds_nn.mean()],
        "Desv. Est.": [y_test.std(), preds_base.std(), preds_nn.std()],
        "Mínimo": [y_test.min(), preds_base.min(), preds_nn.min()],
        "Mediana": [np.median(y_test), np.median(preds_base), np.median(preds_nn)],
        "Máximo": [y_test.max(), preds_base.max(), preds_nn.max()],
    })
    print(stats_df.round(4).to_string(index=False))

    # Guardar predicciones de prueba en CSV
    ruta_preds = f"resultados/t4_predicciones_lgd{sufijo}.csv"
    pd.DataFrame({
        "y_observada": y_test,
        "pred_ridge": preds_base,
        "pred_mlp": preds_nn,
    }).round(5).to_csv(ruta_preds, index=False)
    print(f"\n[OK] Predicciones individuales guardadas en '{ruta_preds}'")

    # 7. Gráfico comparativo de distribuciones (KDE e histogramas normalizados)
    plt.figure(figsize=(9, 5))
    plt.hist(
        y_test,
        bins=30,
        alpha=0.4,
        label="LGD Observada (Real)",
        density=True,
        color="gray",
    )
    plt.hist(
        preds_nn,
        bins=30,
        alpha=0.6,
        label=f"Predicción MLP ({config_nn.get('activacion', 'relu')}, {config_nn['capas']})",
        density=True,
        color="royalblue",
    )
    plt.hist(
        preds_base,
        bins=30,
        alpha=0.45,
        label="Predicción Regresión Ridge",
        density=True,
        color="darkorange",
    )
    plt.title("Comparación de Distribuciones de LGD (Partición de Prueba)")
    plt.xlabel("Loss Given Default (LGD)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    ruta_figura = f"figuras/t4_distribucion_lgd{sufijo}.png"
    plt.savefig(ruta_figura, dpi=130)
    plt.close()
    print(f"[OK] Gráfico de distribuciones actualizado en '{ruta_figura}'")

    # 8. Cálculo de la Pérdida Esperada: E[L] = PD * LGD * EAD
    if "funded_amnt" not in test.columns:
        raise KeyError("La columna 'funded_amnt' (EAD) no está en el conjunto de prueba.")
    ead_values = test["funded_amnt"].values

    # Inferencia de PD usando el modelo de Parte 1
    ruta_pd = "../Parte_1/modelos/red_neuronal.pt"
    if not os.path.exists(ruta_pd):
        ruta_pd = "../Parte_1/modelos/red_neuronal_demo.pt"

    if os.path.exists(ruta_pd) and os.path.exists("../Parte_1/modelos/regresion_logistica.joblib"):
        prep_pd = joblib.load("../Parte_1/modelos/regresion_logistica.joblib")["preprocesador"]
        modelo_pd, _ = cargar_red_pd(ruta_pd)
        X_test_pd = prep_pd.transform(X_test_raw)
        pd_preds = predecir_pd(modelo_pd, X_test_pd)
    else:
        print("[Aviso] Modelo de PD de Parte 1 no encontrado; usando tasa base de referencia (0.20) para E[L].")
        pd_preds = np.full(len(preds_nn), 0.20)

    # Pérdida Esperada individual
    expected_loss = pd_preds * preds_nn * ead_values

    ruta_el = f"resultados/t5_perdida_esperada{sufijo}.csv"
    pd.DataFrame({
        "PD": pd_preds,
        "LGD_Predicha": preds_nn,
        "EAD": ead_values,
        "Expected_Loss": expected_loss,
    }).round(4).to_csv(ruta_el, index=False)

    print(
        f"[OK] Pérdida Esperada calculada exitosamente en '{ruta_el}'. "
        f"Promedio E[L]: ${expected_loss.mean():,.2f} USD por crédito."
    )


def main():
    parser = argparse.ArgumentParser(description="Tarea 4 y 5 (Parte 2): Evaluación LGD y Pérdida Esperada E[L]")
    parser.add_argument("--demo", action="store_true", help="usar modelos y salidas en modo demostración")
    args = parser.parse_args()
    evaluar_modelos_y_calcular_el(demo=args.demo)


if __name__ == "__main__":
    main()

