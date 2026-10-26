# Importamos os para crear carpetas de salida de resultados y figuras
import os

# Importamos sys para la gestion de rutas
import sys

# Importamos joblib para cargar los modelos serializados
import joblib

# Importamos matplotlib para generar la grafica comparativa de distribuciones
import matplotlib.pyplot as plt

# Importamos numpy para operaciones matriciales
import numpy as np

# Importamos pandas para la exportacion de tablas
import pandas as pd

# Importamos las metricas de evaluacion MAE y RMSE
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Importamos PyTorch para cargar los pesos de la red neuronal
import torch

# Añadimos la carpeta Parte_1 a la ruta de modulos
sys.path.append("../Parte_1")

# Importamos la particion temporal unificada
from preprocesamiento import partir

# Importamos la arquitectura de la red neuronal de LGD
from red_neuronal_lgd import RedNeuronalLGD


def evaluar_modelos_y_calcular_el():
    # Cargar la base procesada de LGD
    df_lgd = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])

    # Extraer únicamente la partición de prueba (test) desde el diccionario devuelto por partir()
    partes = partir(df_lgd)
    test = partes["prueba"]

    # Cargar el preprocesador entrenado en la regresion lineal
    prep_lgd = joblib.load("modelos/preprocesador_lgd.joblib")

    # Separar variables predictoras (X) y objetivo (y)
    X_test_raw = test.drop(columns=["target_lgd", "issue_d"])
    y_test = test["target_lgd"].values

    # Transformar el conjunto de prueba
    X_test = prep_lgd.transform(X_test_raw)

    # Convertir a arreglo NumPy si el preprocesador retorna un DataFrame de pandas
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # 1. Generar predicciones del modelo de referencia (Regresion Ridge)
    modelo_base = joblib.load("modelos/regresion_lineal.joblib")
    preds_base = modelo_base.predict(X_test)

    # 2. Generar predicciones de la Red Neuronal (MLP)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    modelo_nn = RedNeuronalLGD(input_dim=X_test.shape[1], capas=[64, 32])
    modelo_nn.load_state_dict(torch.load("modelos/red_neuronal_lgd.pt"))
    modelo_nn.eval()

    # Inferencia con PyTorch sin calculo de gradientes
    with torch.no_grad():
        preds_nn = modelo_nn(X_test_t).cpu().numpy().flatten()

    # Calcular metricas finales en test
    mae_base = mean_absolute_error(y_test, preds_base)
    rmse_base = root_mean_squared_error(y_test, preds_base)
    mae_nn = mean_absolute_error(y_test, preds_nn)
    rmse_nn = root_mean_squared_error(y_test, preds_nn)

    # 3. Generar y guardar gráfico de distribuciones comparativas (KDE / Histogramas)
    os.makedirs("figuras", exist_ok=True)
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
        label="Predicción MLP (Red Neuronal)",
        density=True,
        color="blue",
    )
    plt.hist(
        preds_base,
        bins=30,
        alpha=0.5,
        label="Predicción Regresión Lineal",
        density=True,
        color="orange",
    )
    plt.title("Comparación de Distribuciones de LGD (Test Set)")
    plt.xlabel("Loss Given Default (LGD)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figuras/t4_distribucion_lgd.png")
    plt.close()

    # 4. Calculo de la Perdida Esperada E[L] = PD * LGD * EAD
    # Tomamos el monto de exposicion EAD desde loan_amnt si existe en test
    ead_values = (
        test["loan_amnt"].values if "loan_amnt" in test.columns else 10000.0
    )

    # Cargar las probabilidades de incumplimiento (PD) predichas en la Parte 1 si existen
    pd_path = "../Parte_1/resultados/t5_predicciones_pd.csv"
    if os.path.exists(pd_path):
        df_pd = pd.read_csv(pd_path)
        pd_preds = df_pd["prob_default"].values[: len(preds_nn)]
    else:
        # Probabilidad de referencia predeterminada en caso de evaluacion independiente
        pd_preds = np.full(len(preds_nn), 0.15)

    # Calculamos la Perdida Esperada individual
    expected_loss = pd_preds * preds_nn * ead_values

    # Estructuramos el DataFrame final de Perdida Esperada
    df_el = pd.DataFrame(
        {
            "PD": pd_preds,
            "LGD_Predicha": preds_nn,
            "EAD": ead_values,
            "Expected_Loss": expected_loss,
        }
    )

    # Guardamos los resultados de Perdida Esperada en resultados/
    os.makedirs("resultados", exist_ok=True)
    df_el.to_csv("resultados/t5_perdida_esperada.csv", index=False)

    # Imprimimos confirmacion por consola
    print(
        f"[OK] Evaluacion finalizada. MAE Base: {mae_base:.4f} | MAE MLP: {mae_nn:.4f}"
    )
    print(
        "[OK] Figura guardada en figuras/t4_distribucion_lgd.png y E[L] en resultados/t5_perdida_esperada.csv"
    )


# Punto de entrada de ejecucion independiente
if __name__ == "__main__":
    # Ejecutamos la evaluacion y calculo de E[L]
    evaluar_modelos_y_calcular_el() 