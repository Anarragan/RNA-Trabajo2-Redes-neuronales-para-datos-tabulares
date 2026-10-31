"""
Tarea 3 (Parte 2): Red Neuronal MLP para LGD (Pérdida Dado el Incumplimiento).

Uso:
  python red_neuronal_lgd.py                                                  # búsqueda secuencial completa
  python red_neuronal_lgd.py --demo                                           # versión rápida (< 30 s)
  python red_neuronal_lgd.py --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4 # configuración manual directa
  python red_neuronal_lgd.py --config config.json                             # desde archivo de configuración

Opciones por línea de comandos o archivo de configuración:
  --config       ruta a archivo JSON con hiperparámetros
  --capas        neuronas por capa oculta (ej. --capas 64 32)
  --lr           tasa de aprendizaje / learning rate (ej. --lr 0.001)
  --dropout      tasa de regularización dropout (ej. --dropout 0.2)
  --l2           regularización L2 / weight decay (ej. --l2 1e-4)
  --activacion   función de activación: relu, tanh, leaky_relu (ej. --activacion relu)
  --optimizador  optimizador: adam o sgd (ej. --optimizador adam)
  --epocas       máximo de épocas de entrenamiento por semilla (defecto: 40, o 10 con --demo)
  --semillas     número de semillas para entrenamiento final (defecto: 30, o 3 con --demo)
  --paciencia    épocas para parada temprana / early stopping (defecto: 10)
  --sin-busqueda entrena directamente la configuración dada/base sin búsqueda por etapas
  --demo         versión rápida (< 30 s) con 3 semillas para demostración
"""
import argparse
import json
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
            if dropout > 0:
                bloques.append(nn.Dropout(dropout))
            anterior = neuronas

        # Capa lineal final hacia 1 salida escalar
        bloques.append(nn.Linear(anterior, 1))
        # Salida acotada estrictamente en el rango continuo [0, 1]
        bloques.append(nn.Sigmoid())
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)


def contar_parametros(modelo):
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def a_tensor(X, y=None):
    """Convierte DataFrames/arrays a tensores float32 de PyTorch en el dispositivo adecuado."""
    X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=DISPOSITIVO)
    if y is None:
        return X_tensor
    y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32), device=DISPOSITIVO).view(-1, 1)
    return X_tensor, y_tensor


def predecir_lgd(modelo, X):
    """Genera predicciones continuas de LGD en escala [0, 1]."""
    modelo.eval()
    with torch.no_grad():
        return modelo(a_tensor(X)).cpu().numpy().ravel()


def cargar_red_lgd(ruta="modelos/red_neuronal_lgd.pt"):
    """
    Carga el artefacto serializado del modelo y reconstruye dinámicamente la arquitectura.
    """
    guardado = torch.load(ruta, map_location=DISPOSITIVO, weights_only=False)
    config = guardado["config"]
    modelo = RedNeuronalLGD(
        guardado["n_entrada"],
        capas=config["capas"],
        dropout=config.get("dropout", 0.0),
        activacion=config.get("activacion", "relu"),
    )
    modelo.to(DISPOSITIVO).load_state_dict(guardado["pesos"])
    return modelo.eval(), config


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


def probar_configuracion(config, datos, filas, epocas=EPOCAS, paciencia=PACIENCIA):
    """Evalúa una configuración candidata sobre semillas de búsqueda rápida."""
    resultados = [entrenar_una_corrida(config, datos, 100 + s, epocas=epocas, paciencia=paciencia) for s in range(SEMILLAS_BUSQUEDA)]
    fila = {
        **config,
        "loss_mse_val": float(np.mean([r[1] for r in resultados])),
        "epocas": float(np.mean([r[2] for r in resultados])),
        "segundos": float(sum([r[3] for r in resultados])),
        "parametros": contar_parametros(resultados[-1][0]),
    }
    print(
        f"  {len(filas) + 1:>2} | MSE val {fila['loss_mse_val']:.5f} | "
        f"{fila['epocas']:4.0f} épocas | {fila['segundos']:5.1f} s | {config}"
    )
    filas.append(fila)
    return fila


def buscar_hiperparametros(datos, epocas=EPOCAS, paciencia=PACIENCIA):
    """Búsqueda por etapas de la mejor arquitectura e hiperparámetros para LGD."""
    filas = []
    n_configs = 1 + sum(len(op) for _, op in ETAPAS)
    print(f"\nBÚSQUEDA POR ETAPAS LGD ({n_configs} configuraciones, criterio: MSE de validación)")
    mejor = BASE
    mejor_mse = probar_configuracion(mejor, datos, filas, epocas=epocas, paciencia=paciencia)["loss_mse_val"]

    for claves, opciones in ETAPAS:
        for opcion in opciones:
            valores = opcion if len(claves) > 1 else (opcion,)
            candidata = {**mejor, **dict(zip(claves, valores))}
            mse = probar_configuracion(candidata, datos, filas, epocas=epocas, paciencia=paciencia)["loss_mse_val"]
            if mse < mejor_mse:
                mejor, mejor_mse = candidata, mse

    tabla = pd.DataFrame(filas)
    return mejor, float(tabla["segundos"].sum()), tabla


def entrenar_semillas_finales(config, datos, X, y, n_semillas=N_SEMILLAS, epocas=EPOCAS, paciencia=PACIENCIA):
    """Entrena la configuración ganadora sobre N semillas y registra estabilidad."""
    print(f"\nENTRENAMIENTO FINAL LGD ({n_semillas} semillas)")
    registros = []
    mejor_mae_val, mejor_modelo, mejor_semilla, t_total = float("inf"), None, None, 0.0

    for semilla in range(n_semillas):
        modelo, _, ep_usadas, seg = entrenar_una_corrida(config, datos, semilla, epocas=epocas, paciencia=paciencia)
        t_total += seg

        # Evaluamos métricas de regresión (MAE y RMSE) en entrenamiento, validación y prueba
        preds_train = predecir_lgd(modelo, X["entrenamiento"])
        preds_val = predecir_lgd(modelo, X["validacion"])
        preds_test = predecir_lgd(modelo, X["prueba"])

        mae_train = mean_absolute_error(y["entrenamiento"], preds_train)
        rmse_train = root_mean_squared_error(y["entrenamiento"], preds_train)
        mae_val = mean_absolute_error(y["validacion"], preds_val)
        rmse_val = root_mean_squared_error(y["validacion"], preds_val)
        mae_test = mean_absolute_error(y["prueba"], preds_test)
        rmse_test = root_mean_squared_error(y["prueba"], preds_test)

        registros.append({
            "semilla": semilla,
            "MAE_train": mae_train,
            "RMSE_train": rmse_train,
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo", action="store_true", help="versión rápida (< 30 s: 10 épocas y 3 semillas)")
    parser.add_argument("--config", type=str, default=None, help="ruta a archivo JSON con hiperparámetros (ej: config.json)")
    parser.add_argument("--capas", type=int, nargs="+", default=None, help="neuronas por capa oculta (ej: --capas 64 32)")
    parser.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje / learning rate (ej: --lr 0.001)")
    parser.add_argument("--dropout", type=float, default=None, help="tasa de regularización dropout (ej: --dropout 0.2)")
    parser.add_argument("--l2", type=float, default=None, help="regularización L2 / weight decay (ej: --l2 1e-4)")
    parser.add_argument("--activacion", type=str, choices=["relu", "tanh", "leaky_relu"], default=None, help="función de activación")
    parser.add_argument("--optimizador", type=str, choices=["adam", "sgd"], default=None, help="optimizador (adam o sgd)")
    parser.add_argument("--semillas", type=int, default=None, help="semillas para entrenar la red final (defecto: 30, o 3 con --demo)")
    parser.add_argument("--epocas", type=int, default=None, help="máximo de épocas por semilla (defecto: 40, o 10 con --demo)")
    parser.add_argument("--paciencia", type=int, default=PACIENCIA, help=f"épocas para early stopping (defecto: {PACIENCIA})")
    parser.add_argument("--sin-busqueda", action="store_true", help="usar configuración directa sin búsqueda por etapas")
    args = parser.parse_args()

    ruta_config = os.path.abspath(args.config) if args.config else None

    os.chdir(CARPETA)
    t_inicio = time.perf_counter()
    sufijo = "_demo" if args.demo else ""

    # Cargar configuración base o desde JSON
    config = dict(BASE)
    if ruta_config:
        if not os.path.exists(ruta_config):
            sys.exit(f"Error: No se encontró el archivo de configuración '{args.config}'")
        with open(ruta_config, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
            config.update({k: v for k, v in cfg_json.items() if k in config})
            if "epocas" in cfg_json and args.epocas is None:
                args.epocas = cfg_json["epocas"]
            if "semillas" in cfg_json and args.semillas is None:
                args.semillas = cfg_json["semillas"]
            if "paciencia" in cfg_json:
                args.paciencia = cfg_json["paciencia"]

    # Sobreescribir con flags CLI si fueron pasados
    if args.capas is not None:
        config["capas"] = args.capas
    if args.lr is not None:
        config["lr"] = args.lr
    if args.dropout is not None:
        config["dropout"] = args.dropout
    if args.l2 is not None:
        config["l2"] = args.l2
    if args.activacion is not None:
        config["activacion"] = args.activacion
    if args.optimizador is not None:
        config["optimizador"] = args.optimizador

    es_manual = any([ruta_config, args.capas, args.lr, args.dropout, args.l2, args.activacion, args.optimizador, args.sin_busqueda])

    n_semillas = 3 if args.demo else (args.semillas if args.semillas is not None else N_SEMILLAS)
    epocas = 10 if args.demo else (args.epocas if args.epocas is not None else 40)

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

    # 2. Búsqueda o configuración directa
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    if args.demo:
        t_busqueda = 0.0
        print(f"\n[Modo DEMO] Usando configuración: {config} (épocas={epocas}, semillas={n_semillas})")
    elif es_manual:
        t_busqueda = 0.0
        print(f"\n[Configuración manual directa]: {config} (épocas={epocas}, semillas={n_semillas})")
    else:
        config, t_busqueda, tabla_busqueda = buscar_hiperparametros(datos_tensor, epocas=epocas, paciencia=args.paciencia)
        tabla_busqueda.to_csv("resultados/t4_busqueda_lgd.csv", index=False)
        print(f"\nMEJOR CONFIGURACIÓN LGD: {config} (Tiempo búsqueda: {t_busqueda:.1f} s)")

    # 3. Entrenamiento con semillas independientes
    df_semillas, mejor_modelo, mejor_semilla, t_semillas = entrenar_semillas_finales(
        config, datos_tensor, X, y, n_semillas=n_semillas, epocas=epocas, paciencia=args.paciencia
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
        f"[Prueba - {n_semillas} semillas]\n"
        f"MAE:  {df_semillas['MAE_test'].mean():.4f} +/- {df_semillas['MAE_test'].std():.4f} "
        f"(Mejor: {df_semillas['MAE_test'].min():.4f} | Peor: {df_semillas['MAE_test'].max():.4f})\n"
        f"RMSE: {df_semillas['RMSE_test'].mean():.4f} +/- {df_semillas['RMSE_test'].std():.4f} "
        f"(Mejor: {df_semillas['RMSE_test'].min():.4f} | Peor: {df_semillas['RMSE_test'].max():.4f})\n"
    )

    with open(f"resultados/t4_resumen{sufijo}.txt", "w") as f:
        f.write(resumen)

    print("\n" + resumen)
    print(f"[OK] Modelo guardado en '{ruta_modelo}' y resumen en 'resultados/t4_resumen{sufijo}.txt'")


if __name__ == "__main__":
    main()
