"""
Tarea 4: red neuronal (MLP) para la probabilidad de incumplimiento.
Búsqueda aleatoria de hiperparámetros y luego 30 semillas con la mejor configuración.
Guarda modelos/red_neuronal.pt y resultados/t4_*.csv|txt.

    python red_neuronal.py --demo        # rápido
    python red_neuronal.py --busqueda 0 --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4   # sin búsqueda
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from metricas import evaluar
from preprocesamiento import preparar

# Espacio de búsqueda: de aquí se sortean las configuraciones
ESPACIO = {
    "capas": [[32, 16], [64, 32], [128, 64], [128, 64, 32]],   # neuronas de cada capa oculta
    "activacion": ["relu", "tanh"],                            # función de activación de las capas ocultas
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "optimizador": ["adam", "sgd"],                            # SGD con momento 0.9
    "l2": [0.0, 1e-5, 1e-4],                                   # penalización L2 (weight_decay)
}
# La tasa de aprendizaje se sortea según el optimizador: SGD necesita pasos más grandes que Adam
LR = {"adam": [1e-2, 1e-3, 5e-4], "sgd": [1e-1, 3e-2, 1e-2]}
CLAVES = list(ESPACIO) + ["lr"]                                # todo lo que define una configuración
N_COMBINACIONES = int(np.prod([len(v) for k, v in ESPACIO.items() if k != "optimizador"])
                      * sum(len(v) for v in LR.values()))
PACIENCIA, LOTE = 5, 1024   # parada temprana: épocas sin mejora; tamaño del mini-lote
ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh}


def sortear(rng):
    """Sortea una configuración del espacio de búsqueda."""
    c = {k: v[rng.randint(len(v))] for k, v in ESPACIO.items()}
    lrs = LR[c["optimizador"]]
    c["lr"] = lrs[rng.randint(len(lrs))]
    return c


class RedNeuronal(nn.Module):
    """MLP: [Linear -> activación -> Dropout] por cada capa oculta + una neurona de salida.
    Recibe (n, n_entrada) y devuelve (n, 1) LOGITS; la probabilidad es sigmoide(logit)."""

    def __init__(self, n_entrada, capas, dropout, activacion="relu"):
        super().__init__()
        bloques, anterior = [], n_entrada
        for neuronas in capas:
            bloques += [nn.Linear(anterior, neuronas), ACTIVACIONES[activacion](), nn.Dropout(dropout)]
            anterior = neuronas
        bloques.append(nn.Linear(anterior, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def a_tensor(X, y=None):
    """DataFrame/arreglo -> tensores de PyTorch (float32)."""
    X = torch.tensor(np.asarray(X, dtype="float32"))
    return X if y is None else (X, torch.tensor(np.asarray(y, dtype="float32")).view(-1, 1))


def predecir(modelo, X):
    """Devuelve un arreglo numpy con la probabilidad de ser malo de cada fila de X."""
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(a_tensor(X))).numpy().ravel()


def crear_optimizador(modelo, config):
    """Adam o SGD con momento; en ambos, weight_decay suma l2 * pesos al gradiente (penalización L2)."""
    if config["optimizador"] == "sgd":
        return torch.optim.SGD(modelo.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["l2"])
    return torch.optim.Adam(modelo.parameters(), lr=config["lr"], weight_decay=config["l2"])


def entrenar(config, datos, semilla, epocas, paciencia, lote):
    """Entrena un MLP con mini-lotes y parada temprana sobre la pérdida de validación.

    datos = (X_ent, y_ent, X_val, y_val) ya como tensores.
    Devuelve (modelo con los mejores pesos, mejor pérdida de validación, épocas usadas, segundos).
    """
    X_ent, y_ent, X_val, y_val = datos
    torch.manual_seed(semilla)
    modelo = RedNeuronal(X_ent.shape[1], config["capas"], config["dropout"], config["activacion"])
    optimizador = crear_optimizador(modelo, config)
    perdida = nn.BCEWithLogitsLoss()   # sigmoide + entropía cruzada binaria, numéricamente estable

    t0 = time.perf_counter()
    # Se parte de los pesos iniciales: si una configuración diverge (pérdida NaN) no se rompe el programa,
    # simplemente queda con pérdida de validación infinita y nunca gana la búsqueda.
    mejor_val, mejores_pesos, sin_mejora = float("inf"), {k: v.clone() for k, v in modelo.state_dict().items()}, 0
    for epoca in range(1, epocas + 1):
        modelo.train()
        orden = torch.randperm(len(X_ent))             # se barajan las filas en cada época
        for i in range(0, len(orden), lote):
            filas = orden[i:i + lote]
            optimizador.zero_grad()
            perdida(modelo(X_ent[filas]), y_ent[filas]).backward()
            optimizador.step()
        modelo.eval()
        with torch.no_grad():
            val = perdida(modelo(X_val), y_val).item()
        if val < mejor_val:
            mejor_val, sin_mejora = val, 0
            mejores_pesos = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejores_pesos)
    return modelo, mejor_val, epoca, time.perf_counter() - t0


def cargar_red(ruta="modelos/red_neuronal.pt"):
    """Carga un modelo guardado por este script. Devuelve (modelo, config)."""
    guardado = torch.load(ruta, weights_only=False)
    c = guardado["config"]
    modelo = RedNeuronal(guardado["n_entrada"], c["capas"], c["dropout"], c.get("activacion", "relu"))
    modelo.load_state_dict(guardado["pesos"])
    return modelo.eval(), c


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # las rutas son relativas a Parte_1
    p = argparse.ArgumentParser(description="Tarea 4: red neuronal para la probabilidad de incumplimiento")
    p.add_argument("--demo", action="store_true", help="versión rápida: pocos datos, pocas configuraciones y semillas")
    p.add_argument("--busqueda", type=int, default=10, help="configuraciones a probar (0 = usar la de la línea de comandos)")
    p.add_argument("--semillas", type=int, default=30, help="semillas para entrenar la configuración final")
    p.add_argument("--epocas", type=int, default=40, help="máximo de épocas por entrenamiento")
    p.add_argument("--capas", type=int, nargs="+", default=[64, 32], help="neuronas por capa oculta (sin búsqueda)")
    p.add_argument("--activacion", choices=list(ACTIVACIONES), default="relu", help="activación de las capas ocultas (sin búsqueda)")
    p.add_argument("--optimizador", choices=["adam", "sgd"], default="adam", help="optimizador (sin búsqueda)")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    a = p.parse_args()
    a.busqueda = min(a.busqueda, N_COMBINACIONES)  # no se pueden probar más configuraciones que las del espacio
    if a.demo:
        a.busqueda, a.semillas, a.epocas = min(a.busqueda, 3), min(a.semillas, 3), min(a.epocas, 8)
    sufijo = "_demo" if a.demo else ""
    os.makedirs("modelos", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)
    t_total = time.perf_counter()

    # 1. Datos: los mismos de la regresión logística (partición temporal + preprocesamiento) -----
    prep, X, y, partes = preparar()
    if a.demo:   # solo se reduce el entrenamiento; validación y prueba quedan completas
        n_demo = min(20000, len(y["entrenamiento"]))
        muestra = np.random.RandomState(0).choice(len(y["entrenamiento"]), n_demo, replace=False)
        X_ent, y_ent = X["entrenamiento"].iloc[muestra], y["entrenamiento"][muestra]
    else:
        X_ent, y_ent = X["entrenamiento"], y["entrenamiento"]
    datos = (*a_tensor(X_ent, y_ent), *a_tensor(X["validacion"], y["validacion"]))
    print(f"Entrenamiento {len(y_ent):,} | validación {len(y['validacion']):,} | prueba {len(y['prueba']):,} "
          f"| {X_ent.shape[1]} variables de entrada")

    # 2. Búsqueda aleatoria de hiperparámetros ----------------------------------------------------
    config = {"capas": a.capas, "activacion": a.activacion, "dropout": a.dropout,
              "optimizador": a.optimizador, "lr": a.lr, "l2": a.l2}
    filas, t_busqueda = [], 0.0
    if a.busqueda > 0:
        rng = np.random.RandomState(0)
        print(f"\nBÚSQUEDA ALEATORIA ({a.busqueda} de {N_COMBINACIONES} configuraciones posibles, "
              f"criterio: pérdida de validación)")
        probadas = []
        for i in range(a.busqueda):
            c = sortear(rng)
            while c in probadas:   # no se repite una configuración ya probada
                c = sortear(rng)
            probadas.append(c)
            modelo, val, epocas_usadas, seg = entrenar(c, datos, 0, a.epocas, PACIENCIA, LOTE)
            filas.append({**c, "perdida_val": val, "epocas": epocas_usadas, "segundos": seg,
                          "parametros": contar_parametros(modelo)})
            print(f"  {i + 1:>2}/{a.busqueda} | pérdida val {val:.4f} | {epocas_usadas:>2} épocas | {seg:5.1f} s | {c}")
            t_busqueda += seg
        busqueda = pd.DataFrame(filas)
        busqueda.to_csv(f"resultados/t4_busqueda{sufijo}.csv", index=False)
        config = {k: filas[busqueda["perdida_val"].idxmin()][k] for k in CLAVES}
    print(f"\nCONFIGURACIÓN FINAL: {config}")

    # 3. Configuración final con varias semillas ---------------------------------------------------
    print(f"\nENTRENAMIENTO FINAL ({a.semillas} semillas)")
    registros, mejor_auc, mejor_modelo, mejor_semilla, t_semillas = [], -1, None, None, 0.0
    for semilla in range(a.semillas):
        modelo, val, epocas_usadas, seg = entrenar(config, datos, semilla, a.epocas, PACIENCIA, LOTE)
        t_semillas += seg
        for nombre in X:
            m = evaluar(y[nombre], predecir(modelo, X[nombre]))
            registros.append({"semilla": semilla, "particion": nombre, **m, "epocas": epocas_usadas, "segundos": seg})
            if nombre == "validacion":
                auc_val = m["AUC"]
        if auc_val > mejor_auc:
            mejor_auc, mejor_modelo, mejor_semilla = auc_val, modelo, semilla
        print(f"  semilla {semilla:>2} | AUC val {auc_val:.4f} | {epocas_usadas:>2} épocas | {seg:5.1f} s")
    semillas = pd.DataFrame(registros)
    semillas.to_csv(f"resultados/t4_semillas{sufijo}.csv", index=False)
    torch.save({"config": config, "n_entrada": X_ent.shape[1], "pesos": mejor_modelo.state_dict(),
                "columnas": list(prep.columnas_), "semilla": mejor_semilla}, f"modelos/red_neuronal{sufijo}.pt")

    # 4. Resumen (media, desviación, mejor y peor entre semillas) ------------------------------------
    metricas = ["AUC", "Gini", "KS", "Brier"]
    lineas = [f"Configuración: {config}",
              f"Parámetros entrenables: {contar_parametros(mejor_modelo)}",
              f"Presupuesto: {a.busqueda} configuraciones en la búsqueda ({t_busqueda:.0f} s) + "
              f"{a.semillas} semillas ({t_semillas:.0f} s); tiempo total {time.perf_counter() - t_total:.0f} s",
              f"Modelo guardado: semilla {mejor_semilla} (mejor AUC de validación)", ""]
    for nombre in X:
        tabla = semillas[semillas["particion"] == nombre][metricas]
        resumen = pd.DataFrame({"media": tabla.mean(), "desv": tabla.std(), "mejor": tabla.max(), "peor": tabla.min()})
        resumen.loc["Brier", ["mejor", "peor"]] = [tabla["Brier"].min(), tabla["Brier"].max()]  # Brier: menor es mejor
        lineas += [f"[{nombre}]", resumen.round(4).to_string(), ""]
    texto = "\n".join(lineas)
    open(f"resultados/t4_resumen{sufijo}.txt", "w", encoding="utf-8").write(texto)
    print("\n" + texto)


if __name__ == "__main__":
    main()