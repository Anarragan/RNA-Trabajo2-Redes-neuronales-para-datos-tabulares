"""
Tarea 6 (y apoyo a Tarea 7): Scorecard construida sobre la Red Neuronal (modelos/red_neuronal.pt).
No reentrena el perceptrón multicapa: convierte sus logits exactos en puntos a nivel global y 
descompone los puntos por variable usando dos enfoques complementarios:
    1. Descomposición aditiva exacta para cada cliente mediante Gradientes Integrados (Sundararajan et al., 2017),
       donde la suma de los puntos base + los puntos de las variables reproduce el puntaje exacto de la red.
    2. Tabla global interpretable mediante un modelo sustituto lineal (Surrogate Model) ajustado a los logits
       de la red neuronal, permitiendo asignar puntos por +1 DE o por indicador 0/1 como en la scorecard clásica.

Escala (más riesgo = MENOS puntos):
    odds_buenos = (1 - p_malo) / p_malo = exp(-logit_red)
    Puntaje  = Offset + Factor * ln(odds_buenos) = Offset - Factor * logit_red
    Factor   = PDO / ln(2)                      (cada +PDO puntos duplica las odds de buenos)
    Offset   = Base - Factor * ln(odds_ref)     (con odds = odds_ref el puntaje es Base)

Ejecución:
    python scorecard_red_neuronal.py                                  # Base 600, odds 5:1, PDO 20
    python scorecard_red_neuronal.py --base 600 --odds 5 --pdo 20     # parámetros personalizados
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score, roc_curve

from metricas import evaluar
from preprocesamiento import BINARIAS, CATEGORICAS, CON_LOG, CONTEOS, CONTINUAS, preparar
from red_neuronal import a_tensor, cargar_red


def escala(base, odds_ref, pdo):
    """Deriva el Factor y el Offset a partir del puntaje base, las odds de referencia y el PDO."""
    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds_ref)
    return factor, offset


def puntaje_desde_prob(p, factor, offset):
    """Fórmula directa: convierte la probabilidad de incumplimiento p_malo en puntaje."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return offset + factor * np.log((1 - p) / p)


def puntaje_desde_logit(logit, factor, offset):
    """Como ln((1 - p) / p) = -logit, el puntaje es una transformación afín exacta del logit."""
    return offset - factor * logit


def obtener_logits_y_probs(modelo, X_df):
    """Evalúa la red neuronal y devuelve (logits, probabilidades) como arreglos 1D de NumPy."""
    modelo.eval()
    with torch.no_grad():
        x_t = a_tensor(X_df)
        logits_t = modelo(x_t)
        probs_t = torch.sigmoid(logits_t)
    return logits_t.numpy().ravel(), probs_t.numpy().ravel()


def puntos_gradientes_integrados(modelo, X_df, x_ref_arr, factor, offset, pasos=50, lote=4096):
    """
    Descomposición aditiva EXACTA de los puntos de la red neuronal mediante Gradientes Integrados
    (Integrated Gradients - Sundararajan, Taly & Yan, 2017, ICML).

    Por el Teorema Fundamental del Cálculo para integrales de línea:
        logit(x) - logit(x_ref) = sum_j IG_j(x)
    Al multiplicar por (-Factor):
        Puntaje(x) = (Offset - Factor * logit(x_ref)) + sum_j (-Factor * IG_j(x))
                   = puntos_base_rn                   + sum_j puntos_ij

    Devuelve:
        puntos_base (float): Puntaje del perfil de referencia x_ref.
        contrib_puntos (np.ndarray): Matriz (n_clientes, n_variables) con los puntos exactos aportados por cada variable.
    """
    modelo.eval()
    X_arr = np.asarray(X_df, dtype=np.float32)
    x_ref_t = torch.tensor(x_ref_arr, dtype=torch.float32).view(1, -1)

    with torch.no_grad():
        logit_ref = float(modelo(x_ref_t).item())
    puntos_base = offset - factor * logit_ref

    # Regla del punto medio de Riemann en [0, 1] para máxima exactitud numérica
    alphas = torch.linspace(0.5 / pasos, 1.0 - 0.5 / pasos, pasos, dtype=torch.float32).view(pasos, 1, 1)

    n = len(X_arr)
    contrib_list = []

    for ini in range(0, n, lote):
        x_b = torch.tensor(X_arr[ini:ini + lote], dtype=torch.float32)
        delta = x_b - x_ref_t  # (B, d)
        # Interpolación entre x_ref y x_b: dimensión (pasos, B, d)
        x_interp = x_ref_t.unsqueeze(0) + alphas * delta.unsqueeze(0)
        x_flat = x_interp.reshape(-1, x_b.shape[1]).requires_grad_(True)

        salidas = modelo(x_flat)
        grad_flat = torch.autograd.grad(salidas.sum(), x_flat)[0]
        grad_medio = grad_flat.view(pasos, len(x_b), -1).mean(dim=0)  # (B, d)

        ig_logit = (delta * grad_medio).detach().numpy()
        contrib_list.append(-factor * ig_logit)

    return puntos_base, np.vstack(contrib_list)


def puntos_nivel_sustituto(prep, coef_sur, var, valores, factor):
    """Puntos de una variable continua/conteo en valores crudos usando la aproximación del sustituto lineal."""
    x = pd.Series(np.asarray(valores, dtype=float))
    if var in CON_LOG:
        x = np.log1p(x)
    x = x.clip(prep.lim_inf_[var], prep.lim_sup_[var])
    z = (x - prep.medias_[var]) / prep.desv_[var]
    return -factor * coef_sur[var] * z


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Tarea 6: Scorecard construida sobre la Red Neuronal (MLP)")
    p.add_argument("--base", type=float, default=600, help="puntaje base (por defecto: 600)")
    p.add_argument("--odds", type=float, default=5, help="odds de referencia buenos:malos (5 = 5 buenos por cada malo)")
    p.add_argument("--pdo", type=float, default=20, help="puntos para duplicar las odds (por defecto: 20)")
    p.add_argument("--modelo", default="modelos/red_neuronal.pt", help="ruta al archivo .pt de la red neuronal")
    p.add_argument("--pasos-ig", type=int, default=50, help="pasos de integración para Gradientes Integrados")
    a = p.parse_args()

    os.makedirs("resultados", exist_ok=True)
    os.makedirs("figuras", exist_ok=True)

    factor, offset = escala(a.base, a.odds, a.pdo)
    lineas = [
        f"ESCALA (RED NEURONAL): Base = {a.base:g} puntos con odds {a.odds:g}:1 (p_malo = {1 / (1 + a.odds):.3f}) | PDO = {a.pdo:g}",
        f"                       Factor = PDO/ln2 = {factor:.4f} | Offset = Base - Factor*ln(odds) = {offset:.4f}",
    ]

    # 1. Cargar Red Neuronal y datos (mismo preprocesamiento y partición temporal) -----------------------
    modelo, config_rn = cargar_red(a.modelo)
    prep, X, y, partes = preparar()
    cols = list(prep.columnas_)
    lineas.append(f"MODELO BASE: Red Neuronal MLP {config_rn['capas']} | activación={config_rn.get('activacion', 'relu')} "
                  f"| dropout={config_rn['dropout']} | lr={config_rn['lr']}")

    # 2. Logits, probabilidades y puntajes exactos de la Red Neuronal ------------------------------------
    logits, prob, directo, puntaje_logit = {}, {}, {}, {}
    for k in X:
        logits[k], prob[k] = obtener_logits_y_probs(modelo, X[k][cols])
        directo[k] = puntaje_desde_prob(prob[k], factor, offset)
        puntaje_logit[k] = puntaje_desde_logit(logits[k], factor, offset)

    # 3. Descomposición por variable (Enfoque A: Gradientes Integrados + Enfoque B: Sustituto Lineal) -----
    # Perfil de referencia x_ref = 0 (continuas en la media de entrenamiento y binarias/categorías en 0)
    x_ref = np.zeros(len(cols), dtype=np.float32)
    puntos_base_ig, contrib_ig_ent = puntos_gradientes_integrados(
        modelo, X["entrenamiento"][cols], x_ref, factor, offset, pasos=a.pasos_ig
    )
    _, contrib_ig_prueba = puntos_gradientes_integrados(
        modelo, X["prueba"][cols], x_ref, factor, offset, pasos=a.pasos_ig
    )
    suma_ig_prueba = puntos_base_ig + contrib_ig_prueba.sum(axis=1)

    # Modelo sustituto lineal (Surrogate Model) ajustado a los logits de la red en entrenamiento
    sustituto = Ridge(alpha=1.0).fit(X["entrenamiento"][cols].to_numpy(), logits["entrenamiento"])
    r2_sustituto = r2_score(logits["prueba"], sustituto.predict(X["prueba"][cols].to_numpy()))
    coef_sur = pd.Series(sustituto.coef_, index=cols)
    b0_sur = float(sustituto.intercept_)
    puntos_base_sur = offset - factor * b0_sur
    puntos_unidad_sur = -factor * coef_sur

    # Sensibilidad local en el origen y magnitud media de puntos exactos (IG)
    x_cero = torch.zeros((1, len(cols)), dtype=torch.float32, requires_grad=True)
    modelo(x_cero).backward()
    grad_origen = x_cero.grad.detach().numpy().ravel()
    puntos_grad_origen = pd.Series(-factor * grad_origen, index=cols)
    puntos_ig_abs_medio = pd.Series(np.abs(contrib_ig_prueba).mean(axis=0), index=cols)

    tipo = {c: "continua (+1 DE)" for c in CONTINUAS + CONTEOS}
    tipo.update({c: "indicador 0/1" for c in BINARIAS + ["emp_length_nulo"]})
    for var, ref in CATEGORICAS.items():
        tipo.update({f"{var}_{c}": f"categoría vs {ref}" for c in prep.categorias_[var] if c != ref})

    tabla = pd.DataFrame({
        "tipo": pd.Series(tipo),
        "coef_sustituto_rn": coef_sur,
        "razon_odds_aprox": np.exp(coef_sur),
        "puntos": puntos_unidad_sur,
        "puntos_grad_origen": puntos_grad_origen,
        "impacto_medio_abs_pts_IG": puntos_ig_abs_medio,
    }).loc[cols]
    tabla = tabla.reindex(tabla["impacto_medio_abs_pts_IG"].sort_values(ascending=False).index)
    tabla.round(4).to_csv("resultados/t6_rn_puntos_variables.csv", index_label="variable")

    # 4. Verificaciones matemáticas ----------------------------------------------------------------------
    dif_logit_prob = max(np.abs(puntaje_logit[k] - directo[k]).max() for k in X)
    dif_ig_directo = np.abs(suma_ig_prueba - puntaje_logit["prueba"]).max()
    p_ref = 1 / (1 + a.odds)
    p_doble = 1 / (1 + 2 * a.odds)
    lineas += [
        "",
        "VERIFICACIONES MATEMÁTICAS",
        f"  fórmula desde logit vs fórmula desde p_malo: diferencia máxima {dif_logit_prob:.2e} puntos "
        f"({'OK' if dif_logit_prob < 1e-4 else 'REVISAR'})",
        f"  suma de puntos exactos (Gradientes Integrados) vs puntaje directo de la red en prueba: "
        f"diferencia máxima {dif_ig_directo:.4f} puntos ({'OK' if dif_ig_directo < 0.5 else 'REVISAR'})",
        f"  ajuste del sustituto lineal (Surrogate) sobre los logits de la red en prueba: R² = {r2_sustituto:.4f}",
        f"  odds de referencia ({a.odds:g}:1)  -> {puntaje_desde_prob(p_ref, factor, offset):.2f} puntos (debe ser {a.base:g})",
        f"  odds el doble ({2 * a.odds:g}:1)    -> {puntaje_desde_prob(p_doble, factor, offset):.2f} puntos (debe ser {a.base + a.pdo:g})",
        f"  puntos base exactos en x_ref=0 (Offset - Factor*logit(0)) = {puntos_base_ig:.2f} "
        f"(puntos base del sustituto lineal = {puntos_base_sur:.2f}; "
        f"puntaje medio real en entrenamiento = {directo['entrenamiento'].mean():.2f})",
    ]

    # 5. Discriminación del puntaje (debe coincidir exactamente con la de la Red Neuronal) ----------------
    filas = []
    for k in X:
        auc_pun = roc_auc_score(y[k], -directo[k])
        fpr, tpr, _ = roc_curve(y[k], -directo[k])
        m_mod = evaluar(y[k], prob[k])
        filas.append({
            "particion": k,
            "puntaje_medio": directo[k].mean(),
            "puntaje_min": directo[k].min(),
            "puntaje_max": directo[k].max(),
            "AUC_puntaje": auc_pun,
            "AUC_modelo": m_mod["AUC"],
            "KS_puntaje": float(np.max(tpr - fpr)),
            "KS_modelo": m_mod["KS"],
            "Gini_puntaje": 2 * auc_pun - 1,
            "Brier_modelo": m_mod["Brier"],
        })
    disc = pd.DataFrame(filas)
    disc.to_csv("resultados/t6_rn_discriminacion.csv", index=False)
    lineas += ["", "PUNTAJE POR PARTICIÓN Y DISCRIMINACIÓN (RED NEURONAL)", disc.round(4).to_string(index=False)]

    # 6. Bandas de puntaje (cortes fijos con deciles del entrenamiento) y tasa de malos ------------------
    cortes = np.unique(np.quantile(directo["entrenamiento"], np.linspace(0, 1, 11)[1:-1]))
    bandas = []
    for k in X:
        g = np.digitize(directo[k], cortes)  # 0 = banda de menor puntaje (más riesgosa)
        for i in range(len(cortes) + 1):
            m = g == i
            bandas.append({
                "particion": k,
                "banda": i + 1,
                "desde": directo[k][m].min() if m.any() else np.nan,
                "hasta": directo[k][m].max() if m.any() else np.nan,
                "n": int(m.sum()),
                "tasa_malos": y[k][m].mean() if m.any() else np.nan,
                "prob_media": prob[k][m].mean() if m.any() else np.nan,
            })
    bandas = pd.DataFrame(bandas)
    bandas.to_csv("resultados/t6_rn_bandas.csv", index=False)
    mono = {k: bool(np.all(np.diff(bandas[bandas.particion == k]["tasa_malos"].to_numpy()) <= 1e-9)) for k in X}
    lineas += [
        "",
        f"BANDAS DE PUNTAJE (banda 1 = menos puntos = más riesgo; cortes de deciles de entrenamiento: "
        f"{np.round(cortes, 1).tolist()})",
        bandas[bandas.particion == "prueba"].round(3).to_string(index=False),
        f"  tasa de malos decreciente al subir el puntaje: {mono}",
    ]

    # 7. Tabla de puntos por variable e interpretabilidad (Tarea 6 y Tarea 7) ----------------------------
    lineas += [
        "",
        "PUNTOS POR VARIABLE Y ANÁLISIS DE VARIABLES DE RIESGO (ordenado por impacto medio absoluto IG en prueba; "
        "puntos negativos = AUMENTA el riesgo de incumplimiento)",
        tabla.round(2).to_string(),
    ]

    # 8. Puntos por nivel en valores crudos para las variables continuas ---------------------------------
    ent = partes["entrenamiento"]
    niveles = []
    for var in CONTINUAS + CONTEOS:
        qs = [0.10, 0.25, 0.50, 0.75, 0.90]
        vals = np.unique(ent[var].dropna().quantile(qs).to_numpy())
        pts = puntos_nivel_sustituto(prep, coef_sur, var, vals, factor)
        for v, pt in zip(vals, pts):
            niveles.append({"variable": var, "valor": round(float(v), 2), "puntos": round(float(pt), 2)})
    niveles = pd.DataFrame(niveles)
    niveles.to_csv("resultados/t6_rn_puntos_niveles.csv", index=False)
    lineas += [
        "",
        "PUNTOS POR NIVEL EN VARIABLES CONTINUAS (percentiles 10/25/50/75/90 del entrenamiento, valores originales)",
        niveles.pivot_table(
            index="variable",
            columns=niveles.groupby("variable").cumcount(),
            values=["valor", "puntos"],
        ).round(2).to_string(),
    ]

    texto = "\n".join(lineas)
    with open("resultados/t6_rn_scorecard.txt", "w", encoding="utf-8") as f:
        f.write(texto)
    print(texto)

    # 9. Figura: distribución del puntaje en prueba y tasa de malos por banda ----------------------------
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for etiqueta, valor in (("buenos", 0), ("malos", 1)):
        ejes[0].hist(directo["prueba"][y["prueba"] == valor], bins=40, alpha=0.6, density=True, label=etiqueta)
    ejes[0].set(xlabel="puntaje (Red Neuronal)", ylabel="densidad", title="Puntaje RN en prueba (buenos vs malos)")
    ejes[0].legend()
    for k in X:
        b = bandas[bandas.particion == k]
        ejes[1].plot(b["banda"], b["tasa_malos"], marker="o", label=k)
    ejes[1].set(xlabel="banda de puntaje (1 = menos puntos)", ylabel="tasa observada de malos",
                title="Tasa de malos por banda de puntaje (RN)")
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig("figuras/t6_rn_scorecard.png", dpi=130)
    print("\nArchivos generados: resultados/t6_rn_*.csv|txt y figuras/t6_rn_scorecard.png")


if __name__ == "__main__":
    main()
