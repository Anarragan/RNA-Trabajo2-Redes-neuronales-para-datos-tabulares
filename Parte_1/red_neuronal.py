"""
Tarea 4: perceptrón multicapa (MLP) para la probabilidad de incumplimiento.

Uso:
  python red_neuronal.py                                                  # búsqueda secuencial completa
  python red_neuronal.py --demo                                           # modo demo rápido (< 30 s)
  python red_neuronal.py --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4  # configuración manual directa
  python red_neuronal.py --config config.json                             # desde archivo de configuración

Opciones por línea de comandos o archivo de configuración:
  --config       ruta a archivo JSON con hiperparámetros
  --capas        neuronas por capa oculta (ej. --capas 64 32)
  --lr           tasa de aprendizaje / learning rate (ej. --lr 0.001)
  --dropout      tasa de regularización dropout (ej. --dropout 0.2)
  --l2           regularización L2 / weight decay (ej. --l2 1e-4)
  --activacion   función de activación: relu o tanh (ej. --activacion relu)
  --optimizador  optimizador: adam o sgd (ej. --optimizador adam)
  --epocas       máximo de épocas de entrenamiento por semilla (defecto: 60)
  --semillas     número de semillas para entrenamiento final (defecto: 30)
  --paciencia    épocas para parada temprana / early stopping (defecto: 10)
  --sin-busqueda entrena directamente la configuración dada/base sin búsqueda por etapas
  --demo         versión rápida (< 30 s) con 3 semillas para demostración
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from metricas import evaluar
from preprocesamiento import preparar

# ----------------------------- AJUSTES -----------------------------
SEMILLAS_BUSQUEDA = 3     # semillas por configuración (se promedian para que el azar pese menos)
N_SEMILLAS = 30           # semillas de la configuración final
EPOCAS = 60               # máximo de épocas por entrenamiento
PACIENCIA = 10            # parada temprana: épocas seguidas sin mejorar la pérdida de validación
LOTE = 256                # tamaño del mini-lote
REPETICIONES_PERM = 3     # barajadas por variable en la importancia por permutación

# Configuración de partida de la búsqueda
BASE = {"capas": [32, 16], "activacion": "relu", "dropout": 0.2, "l2": 1e-4,
        "optimizador": "adam", "lr": 1e-3}
# Etapas: (claves que se varían, alternativas a probar además del valor actual)
ETAPAS = [
    (("capas",), [[32], [64, 32], [64, 32, 16]]),
    (("activacion",), ["tanh"]),
    (("dropout",), [0.1, 0.3]),
    (("l2",), [1e-5, 1e-3]),
    (("optimizador", "lr"), [("adam", 5e-4), ("adam", 3e-3), ("sgd", 3e-2)]),   # SGD con momento 0.9
]
ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh}
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- MODELO -----------------------------
class RedNeuronal(nn.Module):
    """MLP: [Linear -> activación -> Dropout] por cada capa oculta y una salida (logit)."""

    def __init__(self, n_entrada, capas, dropout, activacion="relu"):
        super().__init__()
        bloques, anterior = [], n_entrada
        act_fn = ACTIVACIONES.get(activacion, nn.ReLU)
        for neuronas in capas:
            bloques += [nn.Linear(anterior, neuronas), act_fn(), nn.Dropout(dropout)]
            anterior = neuronas
        bloques.append(nn.Linear(anterior, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def a_tensor(X, y=None):
    """DataFrame/arreglo -> tensores float32. Con y devuelve (X, y) con y de forma (n, 1)."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=DISPOSITIVO)
    if y is None:
        return X
    return X, torch.as_tensor(np.asarray(y, dtype=np.float32), device=DISPOSITIVO).view(-1, 1)


def predecir(modelo, X):
    """Probabilidad de ser malo para cada fila de X."""
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(a_tensor(X))).cpu().numpy().ravel()


def cargar_red(ruta="modelos/red_neuronal.pt"):
    """Carga un modelo guardado por este script. Devuelve (modelo, config). Lo usan otros scripts."""
    guardado = torch.load(ruta, map_location=DISPOSITIVO, weights_only=False)
    c = guardado["config"]
    modelo = RedNeuronal(guardado["n_entrada"], c["capas"], c["dropout"], c.get("activacion", "relu"))
    modelo.to(DISPOSITIVO).load_state_dict(guardado["pesos"])
    return modelo.eval(), c


# ----------------------------- ENTRENAMIENTO -----------------------------
def entrenar(config, datos, semilla, max_epocas=EPOCAS, paciencia=PACIENCIA):
    """Mini-lotes con parada temprana sobre la pérdida de validación.

    datos = (X_ent, y_ent, X_val, y_val) como tensores. Devuelve (modelo con los mejores pesos,
    mejor pérdida de validación, épocas usadas, segundos, historial [(época, pérdida_ent, pérdida_val)]).
    """
    X_ent, y_ent, X_val, y_val = datos
    torch.manual_seed(semilla)
    modelo = RedNeuronal(X_ent.shape[1], config["capas"], config["dropout"], config.get("activacion", "relu")).to(DISPOSITIVO)
    if config.get("optimizador") == "sgd":
        optimizador = torch.optim.SGD(modelo.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["l2"])
    else:
        optimizador = torch.optim.Adam(modelo.parameters(), lr=config["lr"], weight_decay=config["l2"])
    perdida = nn.BCEWithLogitsLoss()   # entropía cruzada binaria

    t0 = time.perf_counter()
    mejor_val, sin_mejora, historial, mejores_pesos = float("inf"), 0, [], None
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        orden = torch.randperm(len(X_ent), device=DISPOSITIVO)
        acumulada = 0.0
        for i in range(0, len(orden), LOTE):
            filas = orden[i:i + LOTE]
            if len(filas) < 2:
                continue
            optimizador.zero_grad()
            l = perdida(modelo(X_ent[filas]), y_ent[filas])
            l.backward()
            optimizador.step()
            acumulada += l.item() * len(filas)
        modelo.eval()
        with torch.no_grad():
            val = perdida(modelo(X_val), y_val).item()
        historial.append((epoca, acumulada / len(X_ent), val))
        if not np.isfinite(val):            # configuración divergente: no puede ganar la búsqueda
            break
        if val < mejor_val:
            mejor_val, sin_mejora = val, 0
            mejores_pesos = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)
    return modelo, mejor_val, epoca, time.perf_counter() - t0, historial


# ----------------------------- PASOS DEL PROGRAMA -----------------------------
def probar(config, datos, filas, paciencia=PACIENCIA):
    """Entrena una configuración con SEMILLAS_BUSQUEDA semillas y devuelve su fila de resultados."""
    r = [entrenar(config, datos, 100 + s, paciencia=paciencia) for s in range(SEMILLAS_BUSQUEDA)]
    fila = {**config, "perdida_val": float(np.mean([x[1] for x in r])), "desv_val": float(np.std([x[1] for x in r])),
            "epocas": float(np.mean([x[2] for x in r])), "segundos": float(sum(x[3] for x in r)),
            "parametros": contar_parametros(r[-1][0])}
    print(f"  {len(filas) + 1:>2} | pérdida val {fila['perdida_val']:.4f} ± {fila['desv_val']:.4f} | "
          f"{fila['epocas']:4.0f} épocas | {fila['segundos']:5.1f} s | {config}")
    filas.append(fila)
    return fila


def buscar(datos):
    """Búsqueda por etapas. Devuelve (mejor configuración, segundos totales, tabla de resultados)."""
    filas = []
    n = 1 + sum(len(op) for _, op in ETAPAS)
    print(f"\nBÚSQUEDA POR ETAPAS ({n} configuraciones x {SEMILLAS_BUSQUEDA} semillas, "
          f"criterio: pérdida de validación promedio)")
    mejor = BASE
    mejor_perdida = probar(mejor, datos, filas)["perdida_val"]
    for claves, opciones in ETAPAS:
        for opcion in opciones:
            valores = opcion if len(claves) > 1 else (opcion,)
            candidata = {**mejor, **dict(zip(claves, valores))}
            perdida = probar(candidata, datos, filas)["perdida_val"]
            if perdida < mejor_perdida:
                mejor, mejor_perdida = candidata, perdida
    tabla = pd.DataFrame(filas)
    return mejor, float(tabla["segundos"].sum()), tabla


def entrenar_semillas(config, datos, X, y, n_semillas=N_SEMILLAS, max_epocas=EPOCAS, paciencia=PACIENCIA, sufijo=""):
    """Entrena la configuración final con N_SEMILLAS semillas y guarda métricas, probabilidades y curvas."""
    print(f"\nENTRENAMIENTO FINAL ({n_semillas} semillas)")
    registros, probs = [], {nombre: [] for nombre in X}
    mejor_auc, mejor_modelo, mejor_semilla, curvas, t_total = -1, None, None, None, 0.0
    for semilla in range(n_semillas):
        modelo, _, epocas, seg, historial = entrenar(config, datos, semilla, max_epocas=max_epocas, paciencia=paciencia)
        t_total += seg
        if semilla == 0:
            curvas = pd.DataFrame(historial, columns=["epoca", "perdida_ent", "perdida_val"])
        for nombre in X:
            p = predecir(modelo, X[nombre])
            probs[nombre].append(p)
            m = evaluar(y[nombre], p)
            registros.append({"semilla": semilla, "particion": nombre, **m, "epocas": epocas, "segundos": seg})
            if nombre == "validacion":
                auc_val = m["AUC"]
        if auc_val > mejor_auc:
            mejor_auc, mejor_modelo, mejor_semilla = auc_val, modelo, semilla
        print(f"  semilla {semilla:>2} | AUC val {auc_val:.4f} | {epocas:>2} épocas | {seg:5.1f} s")
    semillas = pd.DataFrame(registros)
    semillas.to_csv(f"resultados/t4_semillas{sufijo}.csv", index=False)
    curvas.to_csv(f"resultados/t4_curvas{sufijo}.csv", index=False)
    np.savez_compressed(f"resultados/t4_probs{sufijo}.npz", **{k: np.array(v) for k, v in probs.items()})
    return semillas, mejor_modelo, mejor_semilla, t_total


def importancia_permutacion(modelo, X, y):
    """Caída del AUC al barajar cada columna de X. Una caída <= 0 sugiere una variable sin aporte."""
    rng = np.random.default_rng(0)
    base = evaluar(y, predecir(modelo, X))["AUC"]
    valores, filas = np.asarray(X, dtype=np.float32), []
    for j, nombre in enumerate(X.columns):
        caidas = []
        for _ in range(REPETICIONES_PERM):
            Xp = valores.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            caidas.append(base - evaluar(y, predecir(modelo, Xp))["AUC"])
        filas.append({"variable": nombre, "caida_auc": float(np.mean(caidas)), "desv": float(np.std(caidas))})
    return pd.DataFrame(filas).sort_values("caida_auc", ascending=False).reset_index(drop=True)


def resumir(config, semillas, mejor_modelo, mejor_semilla, n_configs, t_busqueda, t_semillas, t_total):
    """Imprime presupuesto, complejidad y media, desviación, mejor y peor de cada métrica entre semillas."""
    metricas = ["AUC", "Gini", "KS", "Brier"]
    print(f"\nConfiguración: {config}")
    print(f"Parámetros entrenables: {contar_parametros(mejor_modelo)}")
    print(f"Presupuesto: {n_configs} configuraciones x {SEMILLAS_BUSQUEDA} semillas ({t_busqueda:.0f} s) + "
          f"{len(semillas['semilla'].unique())} semillas finales ({t_semillas:.0f} s); tiempo total {t_total:.0f} s")
    print(f"Modelo guardado: semilla {mejor_semilla} (mejor AUC de validación)")
    for nombre in semillas["particion"].unique():
        tabla = semillas[semillas["particion"] == nombre][metricas]
        resumen = pd.DataFrame({"media": tabla.mean(), "desv": tabla.std(), "mejor": tabla.max(), "peor": tabla.min()})
        resumen.loc["Brier", ["mejor", "peor"]] = [tabla["Brier"].min(), tabla["Brier"].max()]  # Brier: menor es mejor
        print(f"\n[{nombre}]\n{resumen.round(4).to_string()}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo", action="store_true", help="versión rápida (< 30 s) con 3 semillas para demostración")
    p.add_argument("--config", type=str, default=None, help="ruta a archivo JSON con hiperparámetros (ej: config.json)")
    p.add_argument("--capas", type=int, nargs="+", default=None, help="neuronas por capa oculta (ej: --capas 64 32)")
    p.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje / learning rate (ej: --lr 0.001)")
    p.add_argument("--dropout", type=float, default=None, help="tasa de regularización dropout (ej: --dropout 0.2)")
    p.add_argument("--l2", type=float, default=None, help="regularización L2 / weight decay (ej: --l2 1e-4)")
    p.add_argument("--activacion", type=str, choices=["relu", "tanh"], default=None, help="función de activación")
    p.add_argument("--optimizador", type=str, choices=["adam", "sgd"], default=None, help="optimizador (adam o sgd)")
    p.add_argument("--epocas", type=int, default=None, help="máximo de épocas por semilla (defecto: 60, o 10 con --demo)")
    p.add_argument("--semillas", type=int, default=None, help="cantidad de semillas para entrenamiento final (defecto: 30, o 3 con --demo)")
    p.add_argument("--paciencia", type=int, default=PACIENCIA, help=f"épocas para early stopping (defecto: {PACIENCIA})")
    p.add_argument("--sin-busqueda", action="store_true", help="entrenar directamente la configuración dada/base sin búsqueda por etapas")
    a = p.parse_args()

    ruta_config = os.path.abspath(a.config) if a.config else None

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("modelos", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)

    sufijo = "_demo" if a.demo else ""
    t0 = time.perf_counter()

    # Cargar configuración base o desde archivo JSON
    config = dict(BASE)
    if ruta_config:
        if not os.path.exists(ruta_config):
            sys.exit(f"Error: No se encontró el archivo de configuración '{a.config}'")
        with open(ruta_config, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
            config.update({k: v for k, v in cfg_json.items() if k in config})
            if "epocas" in cfg_json and a.epocas is None:
                a.epocas = cfg_json["epocas"]
            if "semillas" in cfg_json and a.semillas is None:
                a.semillas = cfg_json["semillas"]
            if "paciencia" in cfg_json:
                a.paciencia = cfg_json["paciencia"]

    # Sobreescribir con flags CLI si fueron pasados
    if a.capas is not None:
        config["capas"] = a.capas
    if a.lr is not None:
        config["lr"] = a.lr
    if a.dropout is not None:
        config["dropout"] = a.dropout
    if a.l2 is not None:
        config["l2"] = a.l2
    if a.activacion is not None:
        config["activacion"] = a.activacion
    if a.optimizador is not None:
        config["optimizador"] = a.optimizador

    es_manual = any([ruta_config, a.capas, a.lr, a.dropout, a.l2, a.activacion, a.optimizador, a.sin_busqueda])

    n_semillas = 3 if a.demo else (a.semillas if a.semillas is not None else N_SEMILLAS)
    max_epocas = 10 if a.demo else (a.epocas if a.epocas is not None else EPOCAS)

    # 1. Datos
    prep, X, y, _ = preparar()
    datos = (*a_tensor(X["entrenamiento"], y["entrenamiento"]), *a_tensor(X["validacion"], y["validacion"]))
    print(f"Dispositivo: {DISPOSITIVO} | entrenamiento {len(y['entrenamiento']):,} | "
          f"validación {len(y['validacion']):,} | prueba {len(y['prueba']):,} | "
          f"{X['entrenamiento'].shape[1]} variables de entrada")

    # 2. Búsqueda o entrenamiento directo
    if a.demo:
        print(f"\nMODO DEMO: evaluando configuración {config} (épocas={max_epocas}, semillas={n_semillas})")
        _, perdida_val, ep_val, t_busqueda, _ = entrenar(config, datos, semilla=0, max_epocas=max_epocas, paciencia=a.paciencia)
        tabla = pd.DataFrame([{
            "etapa": "base_demo",
            **config,
            "perdida_val": round(perdida_val, 4),
            "epocas": ep_val,
            "segundos": round(t_busqueda, 2),
            "parametros": contar_parametros(RedNeuronal(X["entrenamiento"].shape[1], config["capas"], config["dropout"], config.get("activacion", "relu"))),
        }])
        tabla.to_csv("resultados/t4_busqueda_demo.csv", index=False)
        print(f"  Pérdida validación base: {perdida_val:.4f} ({t_busqueda:.1f} s)")
        semillas, mejor_modelo, mejor_semilla, t_semillas = entrenar_semillas(
            config, datos, X, y, n_semillas=n_semillas, max_epocas=max_epocas, paciencia=a.paciencia, sufijo="_demo"
        )
    elif es_manual:
        print(f"\n[Configuración manual directa]: {config} (épocas={max_epocas}, semillas={n_semillas})")
        _, perdida_val, ep_val, t_busqueda, _ = entrenar(config, datos, semilla=0, max_epocas=max_epocas, paciencia=a.paciencia)
        tabla = pd.DataFrame([{
            "etapa": "manual_directa",
            **config,
            "perdida_val": round(perdida_val, 4),
            "epocas": ep_val,
            "segundos": round(t_busqueda, 2),
            "parametros": contar_parametros(RedNeuronal(X["entrenamiento"].shape[1], config["capas"], config["dropout"], config.get("activacion", "relu"))),
        }])
        tabla.to_csv("resultados/t4_busqueda.csv", index=False)
        print(f"  Pérdida validación inicial: {perdida_val:.4f} ({t_busqueda:.1f} s)")
        semillas, mejor_modelo, mejor_semilla, t_semillas = entrenar_semillas(
            config, datos, X, y, n_semillas=n_semillas, max_epocas=max_epocas, paciencia=a.paciencia, sufijo=""
        )
    else:
        config, t_busqueda, tabla = buscar(datos)
        tabla.to_csv("resultados/t4_busqueda.csv", index=False)
        print(f"\nCONFIGURACIÓN FINAL: {config}")
        semillas, mejor_modelo, mejor_semilla, t_semillas = entrenar_semillas(
            config, datos, X, y, n_semillas=n_semillas, max_epocas=max_epocas, paciencia=a.paciencia, sufijo=""
        )

    torch.save({"config": config, "n_entrada": X["entrenamiento"].shape[1], "pesos": mejor_modelo.state_dict(),
                "columnas": list(prep.columnas_), "semilla": mejor_semilla}, f"modelos/red_neuronal{sufijo}.pt")

    # 3. Importancia por permutación y resumen
    imp = importancia_permutacion(mejor_modelo, X["validacion"], y["validacion"])
    imp.to_csv(f"resultados/t4_importancia{sufijo}.csv", index=False)
    print("\nIMPORTANCIA POR PERMUTACIÓN (caída de AUC en validación)\n" + imp.round(4).to_string())
    print(f"Variables sin aporte (caída <= 0): {imp.loc[imp['caida_auc'] <= 0, 'variable'].tolist() or 'ninguna'}")
    resumir(config, semillas, mejor_modelo, mejor_semilla, len(tabla), t_busqueda, t_semillas, time.perf_counter() - t0)


if __name__ == '__main__':
    main()
