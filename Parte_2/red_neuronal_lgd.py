import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

CARPETA = Path(__file__).resolve().parent
if str(CARPETA.parent / "Parte_1") not in sys.path:
    sys.path.append(str(CARPETA.parent / "Parte_1"))

# Importamos la particion temporal unificada
from preprocesamiento import partir

# Hiperparámetros generales
N_SEMILLAS = 30
EPOCAS = 50
PACIENCIA = 10
LOTE = 128
SEMILLAS_BUSQUEDA = 1

# Configuración base de partida
BASE = {
    "capas": [32, 16],
    "activacion": "relu",
    "dropout": 0.2,
    "l2": 1e-4,
    "optimizador": "adam",
    "lr": 1e-3,
}

# Etapas de búsqueda de hiperparámetros adaptadas a regresión LGD
ETAPAS = [
    (("capas",), [[32], [64, 32], [64, 32, 16]]),
    (("activacion",), ["tanh", "leaky_relu"]),
    (("dropout",), [0.1, 0.3, 0.4]),
    (("l2",), [1e-5, 1e-3]),
    (("optimizador", "lr"), [("adam", 5e-4), ("adam", 3e-3), ("sgd", 1e-2)]),
]

ACTIVACIONES = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- MODELO -----------------------------
class RedNeuronalLGD(nn.Module):
    """
    MLP para estimar la LGD:
    Capas ocultas con activación y Dropout.
    Capa de salida con activación Sigmoide para acotar la LGD estrictamente en [0, 1].
    """

    def __init__(self, n_entrada, capas=[32, 16], dropout=0.2, activacion="relu"):
        super().__init__()
        bloques, anterior = [], n_entrada
        act_class = ACTIVACIONES.get(activacion, nn.ReLU)
        for neuronas in capas:
            bloques.append(nn.Linear(anterior, neuronas))
            bloques.append(act_class())
            bloques.append(nn.Dropout(dropout))
            anterior = neuronas
        bloques.append(nn.Linear(anterior, 1))
        bloques.append(nn.Sigmoid())  # Acota LGD en [0, 1]
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)


def contar_parametros(modelo):
    """Retorna el número de parámetros entrenables del modelo."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def a_tensor(X, y=None):
    """Convierte arreglos o DataFrames a tensores en el dispositivo configurado."""
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=DISPOSITIVO)
    if y is None:
        return X_t
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=DISPOSITIVO).view(-1, 1)
    return X_t, y_t


def predecir_lgd(modelo, X):
    """Genera predicciones continuas de LGD acotadas en [0, 1]."""
    modelo.eval()
    with torch.no_grad():
        X_t = a_tensor(X)
        preds = modelo(X_t).cpu().numpy().ravel()
        return np.clip(preds, 0.0, 1.0)


def cargar_red_lgd(ruta=None):
    """Carga el modelo guardado reconstruyendo dinámicamente su arquitectura."""
    if ruta is None:
        ruta = str(CARPETA / "modelos" / "red_neuronal_lgd.pt")
    guardado = torch.load(str(ruta), map_location=DISPOSITIVO, weights_only=False)
    c = guardado["config"]
    modelo = RedNeuronalLGD(
        guardado["n_entrada"],
        capas=c["capas"],
        dropout=c["dropout"],
        activacion=c.get("activacion", "relu"),
    )
    modelo.to(DISPOSITIVO).load_state_dict(guardado["pesos"])
    return modelo.eval(), c


# ----------------------------- ENTRENAMIENTO -----------------------------
def entrenar_una_corrida(config, datos, semilla, epocas=EPOCAS, paciencia=PACIENCIA):
    """
    Entrena el modelo con mini-lotes y parada temprana sobre la pérdida MSE de validación.
    """
    X_ent, y_ent, X_val, y_val = datos
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    modelo = RedNeuronalLGD(
        X_ent.shape[1],
        capas=config["capas"],
        dropout=config["dropout"],
        activacion=config.get("activacion", "relu"),
    ).to(DISPOSITIVO)

    if config.get("optimizador") == "sgd":
        optimizador = optim.SGD(modelo.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["l2"])
    else:
        optimizador = optim.Adam(modelo.parameters(), lr=config["lr"], weight_decay=config["l2"])

    criterio = nn.MSELoss()

    t0 = time.perf_counter()
    mejor_val, sin_mejora, mejores_pesos = float("inf"), 0, None

    for epoca in range(1, epocas + 1):
        modelo.train()
        orden = torch.randperm(len(X_ent), device=DISPOSITIVO)
        for i in range(0, len(orden), LOTE):
            filas = orden[i:i + LOTE]
            if len(filas) < 2:
                continue
            optimizador.zero_grad()
            out = modelo(X_ent[filas])
            loss = criterio(out, y_ent[filas])
            loss.backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            val_loss = criterio(modelo(X_val), y_val).item()

        if not np.isfinite(val_loss):
            break

        if val_loss < mejor_val:
            mejor_val = val_loss
            sin_mejora = 0
            mejores_pesos = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    if mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)

    return modelo, mejor_val, epoca, time.perf_counter() - t0


def probar_configuracion(config, datos, filas, epocas=EPOCAS):
    """Evalúa una configuración candidata en búsqueda."""
    resultados = [entrenar_una_corrida(config, datos, 100 + s, epocas=epocas) for s in range(SEMILLAS_BUSQUEDA)]
    fila = {
        **config,
        "loss_mse_val": float(np.mean([x[1] for x in resultados])),
        "desv_val": float(np.std([x[1] for x in resultados])),
        "epocas": float(np.mean([x[2] for x in resultados])),
        "segundos": float(sum(x[3] for x in resultados)),
        "parametros": contar_parametros(resultados[-1][0]),
    }
    print(
        f"  {len(filas) + 1:>2} | MSE val: {fila['loss_mse_val']:.5f} | "
        f"{fila['epocas']:2.0f} épocas | {fila['segundos']:4.1f} s | {config}"
    )
    filas.append(fila)
    return fila


def buscar_hiperparametros(datos, epocas=EPOCAS):
    """Búsqueda por etapas de la mejor arquitectura e hiperparámetros para LGD."""
    filas = []
    n_configs = 1 + sum(len(op) for _, op in ETAPAS)
    print(f"\nBÚSQUEDA POR ETAPAS LGD ({n_configs} configuraciones, criterio: MSE de validación)")
    mejor = BASE
    mejor_mse = probar_configuracion(mejor, datos, filas, epocas=epocas)["loss_mse_val"]

    for claves, opciones in ETAPAS:
        for opcion in opciones:
            valores = opcion if len(claves) > 1 else (opcion,)
            candidata = {**mejor, **dict(zip(claves, valores))}
            mse = probar_configuracion(candidata, datos, filas, epocas=epocas)["loss_mse_val"]
            if mse < mejor_mse:
                mejor, mejor_mse = candidata, mse

    tabla = pd.DataFrame(filas)
    return mejor, float(tabla["segundos"].sum()), tabla


def entrenar_semillas_finales(config, datos, X, y, n_semillas=N_SEMILLAS, epocas=EPOCAS):
    """Entrena la configuración ganadora sobre N semillas y registra estabilidad."""
    print(f"\nENTRENAMIENTO FINAL LGD ({n_semillas} semillas)")
    registros = []
    mejor_mae_val, mejor_modelo, mejor_semilla, t_total = float("inf"), None, None, 0.0

    for semilla in range(n_semillas):
        modelo, _, ep_usadas, seg = entrenar_una_corrida(config, datos, semilla, epocas=epocas)
        t_total += seg

        # Evaluamos métricas de regresión (MAE y RMSE) en validación y prueba
        preds_val = predecir_lgd(modelo, X["validacion"])
        preds_test = predecir_lgd(modelo, X["prueba"])

        mae_val = mean_absolute_error(y["validacion"], preds_val)
        rmse_val = root_mean_squared_error(y["validacion"], preds_val)
        mae_test = mean_absolute_error(y["prueba"], preds_test)
        rmse_test = root_mean_squared_error(y["prueba"], preds_test)

        registros.append({
            "semilla": semilla,
            "MAE_val": mae_val,
            "RMSE_val": rmse_val,
            "MAE_test": mae_test,
            "RMSE_test": rmse_test,
            "epocas": ep_usadas,
            "segundos": seg,
        })

        if mae_val < mejor_mae_val:
            mejor_mae_val = mae_val
            mejor_modelo = modelo
            mejor_semilla = semilla

        if n_semillas > 5 and semilla % 5 == 0 or n_semillas <= 5:
            print(f"  semilla {semilla:>2} | MAE test {mae_test:.4f} | RMSE test {rmse_test:.4f} | {ep_usadas:>2} épocas | {seg:4.1f} s")

    df_semillas = pd.DataFrame(registros)
    return df_semillas, mejor_modelo, mejor_semilla, t_total


def main():
    os.chdir(CARPETA)
    parser = argparse.ArgumentParser(description="Tarea 3 (Parte 2): Red Neuronal MLP para LGD")
    parser.add_argument("--demo", action="store_true", help="versión rápida (< 30 s: configuración base y 3 semillas)")
    parser.add_argument("--semillas", type=int, default=30, help="semillas para entrenar la red final (defecto: 30)")
    parser.add_argument("--epocas", type=int, default=40, help="máximo de épocas por semilla (defecto: 40)")
    parser.add_argument("--sin-busqueda", action="store_true", help="usar configuración base directamente sin búsqueda")
    args = parser.parse_args()

    t_inicio = time.perf_counter()
    sufijo = "_demo" if args.demo else ""
    n_semillas = 3 if args.demo else args.semillas
    epocas = 10 if args.demo else args.epocas

    # 1. Datos
    df = pd.read_csv("resultados/lgd_procesado.csv", parse_dates=["issue_d"])
    partes = partir(df)

    prep = joblib.load("modelos/preprocesador_lgd.joblib")
    X = {k: prep.transform(partes[k].drop(columns=["target_lgd", "issue_d", "funded_amnt"])) for k in partes}
    y = {k: partes[k]["target_lgd"].values for k in partes}

    datos_tensor = (
        *a_tensor(X["entrenamiento"], y["entrenamiento"]),
        *a_tensor(X["validacion"], y["validacion"]),
    )

    print(
        f"Datos LGD | Entrenamiento: {len(y['entrenamiento']):,} | "
        f"Validación: {len(y['validacion']):,} | Prueba: {len(y['prueba']):,} | "
        f"Variables: {X['entrenamiento'].shape[1]}"
    )

    # 2. Búsqueda de hiperparámetros
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    if args.demo or args.sin_busqueda:
        config = BASE
        t_busqueda = 0.0
        print(f"\n[Modo rápido] Usando configuración base directa: {config}")
    else:
        config, t_busqueda, tabla_busqueda = buscar_hiperparametros(datos_tensor, epocas=epocas)
        tabla_busqueda.to_csv("resultados/t4_busqueda_lgd.csv", index=False)
        print(f"\nMEJOR CONFIGURACIÓN LGD: {config} (Tiempo búsqueda: {t_busqueda:.1f} s)")

    # 3. Entrenamiento con semillas independientes
    df_semillas, mejor_modelo, mejor_semilla, t_semillas = entrenar_semillas_finales(
        config, datos_tensor, X, y, n_semillas=n_semillas, epocas=epocas
    )
    df_semillas.to_csv(f"resultados/t4_semillas_lgd{sufijo}.csv", index=False)

    # 4. Guardar mejor modelo con metadatos completos para reconstrucción dinámica
    ruta_modelo = f"modelos/red_neuronal_lgd{sufijo}.pt"
    torch.save({
        "config": config,
        "n_entrada": X["entrenamiento"].shape[1],
        "pesos": mejor_modelo.state_dict(),
        "semilla": mejor_semilla,
    }, ruta_modelo)

    # 5. Resumen
    num_params = contar_parametros(mejor_modelo)
    t_total = time.perf_counter() - t_inicio
    resumen = (
        f"--- RESUMEN MODELO LGD (MLP) ---\n"
        f"Configuracion: {config}\n"
        f"Parametros Entrenables: {num_params}\n"
        f"Semillas evaluadas: {n_semillas} (Mejor semilla: {mejor_semilla})\n"
        f"Tiempo busqueda: {t_busqueda:.1f} s | Tiempo semillas: {t_semillas:.1f} s | Total: {t_total:.1f} s\n\n"
        f"[Prueba - 30 semillas]\n"
        f"MAE:  {df_semillas['MAE_test'].mean():.4f} +/- {df_semillas['MAE_test'].std():.4f} "
        f"(Mejor: {df_semillas['MAE_test'].min():.4f} | Peor: {df_semillas['MAE_test'].max():.4f})\n"
        f"RMSE: {df_semillas['RMSE_test'].mean():.4f} +/- {df_semillas['RMSE_test'].std():.4f} "
        f"(Mejor: {df_semillas['RMSE_test'].min():.4f} | Peor: {df_semillas['RMSE_test'].max():.4f})\n"
    )

    with open("resultados/t4_resumen.txt", "w") as f:
        f.write(resumen)

    print("\n" + resumen)
    print(f"[OK] Modelo guardado en '{ruta_modelo}' y resumen en 'resultados/t4_resumen.txt'")


if __name__ == "__main__":
    main()
