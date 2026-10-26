# Importamos os para verificacion y creacion de carpetas
import os

# Importamos sys para manejar las rutas de importacion
import sys

# Importamos joblib para cargar el preprocesador oficial
import joblib

# Importamos numpy para el calculo de metricas agregadas (media, desv std)
import numpy as np

# Importamos pandas para la manipulacion de las particiones de datos
import pandas as pd

# Importamos las metricas MAE y RMSE de scikit-learn para regresion
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Importamos PyTorch para la construccion y entrenamiento de la red neuronal
import torch

# Importamos el modulo de capas y funciones de perdida de PyTorch
import torch.nn as nn

# Importamos los optimizadores de PyTorch
import torch.optim as optim

# Añadimos la carpeta Parte_1 a la ruta de modulos
sys.path.append("../Parte_1")

# Importamos la funcion de particion temporal unificada
from preprocesamiento import partir


# Definicion de la arquitectura MLP para la LGD
class RedNeuronalLGD(nn.Module):

    # Constructor que recibe dimension de entrada, capas ocultas y dropout
    def __init__(self, input_dim, capas=[64, 32], dropout=0.2):
        # Inicializamos la clase base nn.Module
        super(RedNeuronalLGD, self).__init__()

        # Lista vacia para almacenar la secuencia de capas
        layers = []

        # Entrada inicial a la primera capa
        in_features = input_dim

        # Construimos iterativamente las capas ocultas
        for units in capas:
            # Capa lineal conectada
            layers.append(nn.Linear(in_features, units))
            # Normalizacion por lote para estabilizar gradientes
            layers.append(nn.BatchNorm1d(units))
            # Activacion ReLU
            layers.append(nn.ReLU())
            # Capa de dropout para prevenir sobreajuste
            layers.append(nn.Dropout(dropout))
            # Actualizamos la dimension de entrada para la siguiente capa
            in_features = units

        # Capa de salida lineal de 1 sola neurona (regresion)
        layers.append(nn.Linear(in_features, 1))

        # Activacion de salida Sigmoide para acotar la LGD estrictamente en [0, 1]
        layers.append(nn.Sigmoid())

        # Ensamblamos la secuencia completa en un modulo Sequential
        self.net = nn.Sequential(*layers)

    # Metodo de pase hacia adelante (forward pass)
    def forward(self, x):
        # Evaluamos la entrada x a traves de la red secuencial
        return self.net(x)


def entrenar_30_semillas(num_semillas=30):
    # Cargar la base procesada de LGD
    df = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])

    # Aplicamos la misma particion temporal (retorna diccionario)
    partes = partir(df)
    train = partes["entrenamiento"]
    val = partes["validacion"]

    # Cargar el preprocesador entrenado en la regresion lineal
    prep = joblib.load("modelos/preprocesador_lgd.joblib")

    # Separar variables predictoras (X) y objetivo (y)
    X_train_raw = train.drop(columns=["target_lgd", "issue_d"])
    y_train = train["target_lgd"].values

    X_val_raw = val.drop(columns=["target_lgd", "issue_d"])
    y_val = val["target_lgd"].values

    # Transformar las particiones con el preprocesador oficial
    X_train = prep.transform(X_train_raw)
    X_val = prep.transform(X_val_raw)

    # Aseguramos conversión explicita a NumPy ndarray por si transform devuelve un DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values


    # Convertir las matrices de NumPy a tensores de PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # Obtener el numero de variables de entrada
    input_dim = X_train.shape[1]

    # Listas para almacenar las metricas de cada semilla
    maes, rmses = [], []

    # Variables para rastrear el mejor modelo segun MAE
    best_mae = float("inf")
    best_model = None

    # Mensaje inicial del benchmark
    print(f"Ejecutando evaluacion sobre {num_semillas} semillas...")

    # Bucle de entrenamiento sobre N semillas independientes
    for seed in range(num_semillas):
        # Fijar la semilla de PyTorch
        torch.manual_seed(seed)
        # Fijar la semilla de NumPy
        np.random.seed(seed)

        # Instanciar la red neuronal LGD
        model = RedNeuronalLGD(input_dim=input_dim, capas=[64, 32], dropout=0.2)

        # Configurar optimizador AdamW
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Configurar funcion de perdida Error Cuadratico Medio (MSE)
        criterion = nn.MSELoss()

        # Colocar la red en modo de entrenamiento
        model.train()

        # Bucle de entrenamiento por 30 epocas
        for epoch in range(30):
            # Limpiar gradientes del paso anterior
            optimizer.zero_grad()
            # Pase hacia adelante
            out = model(X_train_t)
            # Calculo de la perdida
            loss = criterion(out, y_train_t)
            # Retropropagacion de gradientes
            loss.backward()
            # Actualizacion de pesos
            optimizer.step()

        # Colocar la red en modo de evaluacion
        model.eval()

        # Desactivar calculo de gradientes para inferencia
        with torch.no_grad():
            # Predecir sobre el conjunto de validacion
            preds = model(X_val_t).cpu().numpy().flatten()

        # Calcular MAE y RMSE de esta semilla
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)

        # Guardar metricas en las listas
        maes.append(mae)
        rmses.append(rmse)

        # Si encontramos una semilla con mejor MAE, guardamos la red
        if mae < best_mae:
            best_mae = mae
            best_model = model

    # Crear carpeta modelos si no existe
    os.makedirs("modelos", exist_ok=True)

    # Guardar el estado del mejor modelo entrenado
    torch.save(best_model.state_dict(), "modelos/red_neuronal_lgd.pt")

    # Calcular cantidad de parametros entrenables
    num_params = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )

    # Formatear el texto de resumen de resultados
    resumen = (
        f"MAE Promedio: {np.mean(maes):.4f} +/- {np.std(maes):.4f}\n"
        f"MAE Mejor: {np.min(maes):.4f} | Peor: {np.max(maes):.4f}\n"
        f"RMSE Promedio: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}\n"
        f"Parametros Entrenables: {num_params}\n"
    )

    # Crear carpeta resultados si no existe
    os.makedirs("resultados", exist_ok=True)

    # Guardar el resumen explicativo en resultados/t4_resumen.txt
    with open("resultados/t4_resumen.txt", "w") as f:
        f.write(resumen)

    # Mostrar confirmacion por consola
    print(
        f"[OK] Red Neuronal entrenada sobre {num_semillas} semillas. Resumen guardado en resultados/t4_resumen.txt"
    )


# Punto de entrada de ejecucion independiente
if __name__ == "__main__":
    # Ejecutamos las 30 semillas
    entrenar_30_semillas(num_semillas=30)