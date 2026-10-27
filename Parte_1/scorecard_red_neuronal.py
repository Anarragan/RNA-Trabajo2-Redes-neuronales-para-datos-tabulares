"""
Tarea 6: scorecard sobre la red neuronal (modelos/red_neuronal.pt). No reentrena nada.

Idea en tres pasos:
  1. La red da un logit z = ln(p / (1 - p)), con p = probabilidad de ser malo.
  2. El logit se convierte en puntaje (más puntos = menos riesgo):
         odds_buenos = (1 - p) / p = exp(-z)
         Puntaje = Offset + Factor * ln(odds_buenos) = Offset - Factor * z
         Factor  = PDO / ln(2)               (cada +PDO puntos duplica las odds de buenos)
         Offset  = Base - Factor * ln(odds_ref)   (con odds = odds_ref el puntaje es Base)
  3. El puntaje de cada persona se reparte entre sus variables con Gradientes Integrados
     (Sundararajan, Taly y Yan, 2017). La referencia es el cliente promedio de entrenamiento:
         Puntaje = puntos_base + suma de puntos por variable
     "puntos_base" es el puntaje del cliente promedio; cada variable suma o resta frente a él.

    python scorecard_red_neuronal.py                              # Base 600, odds 5:1, PDO 20
    python scorecard_red_neuronal.py --base 600 --odds 5 --pdo 20
    python scorecard_red_neuronal.py --recalibrar                 # ajusta el nivel con la validación

Las funciones de abajo se reutilizan en la aplicación web.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from preprocesamiento import preparar
from red_neuronal import a_tensor, cargar_red


def desplazamiento_logit(z_val, y_val):
    """Constante d tal que la probabilidad media de validación, sigmoide(z + d), iguale su tasa observada de malos."""
    from scipy.optimize import brentq
    return brentq(lambda d: (1 / (1 + np.exp(-(z_val + d)))).mean() - np.mean(y_val), -5, 5)


def escala(base, odds_ref, pdo):
    factor = pdo / np.log(2)
    return factor, base - factor * np.log(odds_ref)


def logits(modelo, X):
    """Logit de la red para cada fila de X (DataFrame o arreglo)."""
    modelo.eval()
    with torch.no_grad():
        return modelo(a_tensor(X)).numpy().ravel()


def puntos_por_variable(modelo, X, x_ref, factor, pasos=50, lote=4096):
    """Matriz (filas x variables) de puntos por variable, con Gradientes Integrados.
    La suma de cada fila es (puntaje de la persona) - (puntaje del cliente de referencia)."""
    modelo.eval()
    X = np.asarray(X, dtype="float32")
    ref = torch.tensor(np.asarray(x_ref, dtype="float32")).view(1, -1)
    alfas = torch.linspace(0.5 / pasos, 1 - 0.5 / pasos, pasos).view(pasos, 1, 1)   # regla del punto medio
    salida = []
    for i in range(0, len(X), lote):
        x = torch.tensor(X[i:i + lote])
        dif = x - ref
        x_camino = (ref.unsqueeze(0) + alfas * dif.unsqueeze(0)).reshape(-1, x.shape[1]).requires_grad_(True)
        grad = torch.autograd.grad(modelo(x_camino).sum(), x_camino)[0].view(pasos, len(x), -1).mean(0)
        salida.append(-factor * (dif * grad).detach().numpy())   # el signo "-" pasa de logit a puntos
    return np.vstack(salida)


def percentil(puntaje, referencia_ordenada):
    """% de la población de referencia con un puntaje menor o igual al dado."""
    return 100 * np.searchsorted(referencia_ordenada, puntaje, side="right") / len(referencia_ordenada)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # las rutas son relativas a Parte_1
    p = argparse.ArgumentParser(description="Tarea 6: scorecard sobre la red neuronal")
    p.add_argument("--base", type=float, default=600, help="puntaje base")
    p.add_argument("--odds", type=float, default=5, help="odds de referencia buenos:malos (5 = 5 buenos por cada malo)")
    p.add_argument("--pdo", type=float, default=20, help="puntos para duplicar las odds")
    p.add_argument("--modelo", default="modelos/red_neuronal.pt")
    p.add_argument("--recalibrar", action="store_true", help="desplaza el nivel de las probabilidades con la validación")
    a = p.parse_args()

    # 1. Escala, red y datos ----------------------------------------------------------------------
    factor, offset = escala(a.base, a.odds, a.pdo)
    print(f"ESCALA: base {a.base:g} puntos con odds {a.odds:g}:1 (p_malo = {1 / (1 + a.odds):.3f}) | PDO {a.pdo:g}"
          f" | Factor = {factor:.3f} | Offset = {offset:.3f}")
    modelo, config = cargar_red(a.modelo)
    prep, X, y, _ = preparar()
    cols = list(prep.columnas_)
    z = {k: logits(modelo, X[k][cols]) for k in X}

    # 2. Puntaje por persona ------------------------------------------------------------------------
    delta = 0.0
    if a.recalibrar:   # desplazar el logit una constante = recalibrar el intercepto; el orden no cambia
        delta = desplazamiento_logit(z["validacion"], y["validacion"])
        print(f"RECALIBRADO: logit desplazado {delta:+.3f} (con validación; AUC, Gini y KS no cambian)")
    z = {k: v + delta for k, v in z.items()}
    prob = {k: 1 / (1 + np.exp(-v)) for k, v in z.items()}
    puntaje = {k: offset - factor * v for k, v in z.items()}

    print("\nVERIFICACIONES")
    for etiqueta, o in (("odds de referencia", a.odds), ("odds el doble", 2 * a.odds)):
        pm = 1 / (1 + o)
        print(f"  {etiqueta} ({o:g}:1) -> {offset + factor * np.log((1 - pm) / pm):.2f} puntos")
    auc = {k: (roc_auc_score(y[k], prob[k]), roc_auc_score(y[k], -puntaje[k])) for k in X}
    print("  AUC modelo vs AUC puntaje (deben ser iguales): "
          + " | ".join(f"{k} {m:.4f}/{s:.4f}" for k, (m, s) in auc.items()))

    # 3. Puntos por variable: Gradientes Integrados frente al cliente promedio de entrenamiento ----------
    x_ref = X["entrenamiento"][cols].mean().to_numpy()
    puntos_base = offset - factor * (float(logits(modelo, x_ref.reshape(1, -1))[0]) + delta)
    puntos = puntos_por_variable(modelo, X["prueba"][cols], x_ref, factor)
    dif = np.abs(puntos_base + puntos.sum(axis=1) - puntaje["prueba"]).max()
    print(f"  puntos base + suma de puntos por variable vs puntaje de la red (prueba): diferencia máxima {dif:.3f} puntos")
    print(f"\nPUNTOS BASE (cliente promedio de entrenamiento) = {puntos_base:.1f}")

    valores = X["prueba"][cols].to_numpy()
    direccion = [np.corrcoef(valores[:, j], puntos[:, j])[0, 1] if valores[:, j].std() > 0 else np.nan
                 for j in range(len(cols))]
    tabla = pd.DataFrame({"impacto_pts": np.abs(puntos).mean(axis=0), "direccion": direccion}, index=cols)
    tabla = tabla.sort_values("impacto_pts", ascending=False)
    print("\nPUNTOS POR VARIABLE EN PRUEBA (ordenado por impacto)")
    print("  impacto_pts = cuántos puntos mueve la variable, en promedio (valor absoluto)")
    print("  direccion   = correlación entre el valor de la variable y sus puntos: "
          "negativa = valores más altos restan puntos (más riesgo); positiva = suman puntos")
    print(tabla.round(2).to_string())
    os.makedirs("resultados", exist_ok=True)
    tabla.round(4).to_csv("resultados/t6_puntos_variables.csv", index_label="variable")

    # 4. Bandas de puntaje: la tasa de malos debe bajar al subir el puntaje ------------------------------------
    cortes = np.unique(np.quantile(puntaje["entrenamiento"], np.linspace(0, 1, 11)[1:-1]))
    filas = {}
    for k in ("validacion", "prueba"):
        g = np.digitize(puntaje[k], cortes)   # 0 = banda de menos puntos (más riesgosa)
        filas[k] = pd.DataFrame({"tasa_malos": [y[k][g == i].mean() for i in range(len(cortes) + 1)],
                                 "n": [(g == i).sum() for i in range(len(cortes) + 1)]})
    bandas = pd.DataFrame({"desde_pts": [puntaje["prueba"][np.digitize(puntaje["prueba"], cortes) == i].min()
                                         for i in range(len(cortes) + 1)],
                           "tasa_malos_val": filas["validacion"]["tasa_malos"],
                           "tasa_malos_prueba": filas["prueba"]["tasa_malos"], "n_prueba": filas["prueba"]["n"]})
    bandas.index += 1
    print("\nBANDAS DE PUNTAJE (1 = menos puntos; cortes = deciles de entrenamiento)")
    print(bandas.round(3).to_string())
    for k in ("validacion", "prueba"):
        print(f"  tasa de malos decreciente al subir el puntaje en {k}: "
              f"{bool(np.all(np.diff(filas[k]['tasa_malos']) <= 1e-9))}")

    # 5. Todo lo que necesita la aplicación web, en un solo archivo -----------------------------------------------
    joblib.dump({"preprocesador": prep, "base": a.base, "odds": a.odds, "pdo": a.pdo, "factor": factor,
                 "offset": offset, "delta": delta, "x_ref": x_ref, "puntos_base": puntos_base,
                 "puntajes_referencia": np.sort(puntaje["entrenamiento"]).astype("float32"),
                 "modelo_red": a.modelo}, "modelos/scorecard.joblib")
    print("\nGuardado: modelos/scorecard.joblib (para la app) y resultados/t6_puntos_variables.csv")


if __name__ == "__main__":
    main()
    