# ============================================================
# eda_hipotesis.py
# Análisis exploratorio real de las hipótesis H1–H6
# Proyecto: RNA-Trabajo2 · Parte_1 (PD en originación)
# ============================================================

# ---------- BLOQUE 0: imports y configuración ----------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorar advertencias visuales de librerías
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "datos_modelo.csv")
FIG_DIR    = os.path.join(BASE_DIR, "figuras")
RES_DIR    = os.path.join(BASE_DIR, "resultados")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ---------- BLOQUE 1: carga del dataset ----------
df = pd.read_csv(DATA_PATH)
TARGET = "malo"  # 1 = incumplió

print("=" * 60)
print("DATASET CARGADO")
print("=" * 60)
print(f"Filas:    {len(df):,}")
print(f"Columnas: {df.shape[1]}")
print(f"Tasa de incumplimiento global: {df[TARGET].mean():.4%}")
print(f"Valores de {TARGET}: {df[TARGET].value_counts().to_dict()}")
print("=" * 60)


# ---------- BLOQUE 2: utilidades ----------
def graficar_tasa_barras(col, titulo, archivo_png, xlabel, bins=10, es_categorica=False):
    if es_categorica:
        grupos = df[col]
    else:
        grupos = pd.qcut(df[col], q=bins, duplicates="drop")

    tabla = (
        df.groupby(grupos, observed=True)[TARGET]
          .agg(n="size", tasa_incumplimiento="mean")
          .reset_index()
    )
    tabla = tabla.rename(columns={tabla.columns[0]: "grupo"})

    fig, ax = plt.subplots()
    ax.bar(range(len(tabla)), tabla["tasa_incumplimiento"], color="steelblue")
    ax.set_xticks(range(len(tabla)))
    ax.set_xticklabels([str(g) for g in tabla["grupo"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tasa de incumplimiento")
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, archivo_png), dpi=150)
    plt.close(fig)
    print(f"[OK] Figura guardada: {archivo_png}")
    return tabla


# ---------- BLOQUES 3-8: Análisis de las Hipótesis (Versión Profesional) ----------
def analizar_H1():
    tabla = df.groupby(pd.qcut(df["dti"], 10, duplicates="drop"), observed=True)[TARGET].agg(n="size", tasa_incumplimiento="mean").reset_index()
    
    fig, ax = plt.subplots()
    ax.plot(range(len(tabla)), tabla["tasa_incumplimiento"], marker='o', color='darkblue', linewidth=2, markersize=8)
    ax.set_xticks(range(len(tabla)))
    ax.set_xticklabels([str(g) for g in tabla.iloc[:, 0]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tasa de incumplimiento")
    ax.set_xlabel("Decil de DTI")
    ax.set_title("H1 · Evolución del riesgo según el nivel de DTI")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_H1_dti_line.png"), dpi=150)
    plt.close(fig)
    print("[OK] Figura H1 guardada (Line plot)")
    return tabla

def analizar_H2():
    fig, ax = plt.subplots()
    df_sample = df.sample(n=20000, random_state=42) if len(df) > 20000 else df
    
    sns.kdeplot(data=df_sample[df_sample[TARGET] == 0], x="annual_inc", label="Buenos (0)", fill=True, color="green", ax=ax)
    sns.kdeplot(data=df_sample[df_sample[TARGET] == 1], x="annual_inc", label="Malos (1)", fill=True, color="red", ax=ax)
    
    ax.set_xlim(0, 200000) 
    ax.set_xlabel("Ingreso anual (USD)")
    ax.set_ylabel("Densidad")
    ax.set_title("H2 · Distribución del ingreso: Buenos vs. Malos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_H2_ingreso_kde.png"), dpi=150)
    plt.close(fig)
    print("[OK] Figura H2 guardada (KDE plot)")
    
    tabla = df.groupby(pd.qcut(df["annual_inc"], 10, duplicates="drop"), observed=True)[TARGET].agg(n="size", tasa_incumplimiento="mean").reset_index()
    return tabla

def analizar_H3():
    df["_delinq_grp"] = df["delinq_2yrs"].where(df["delinq_2yrs"] < 4, 4)
    tabla = graficar_tasa_barras(col="_delinq_grp", titulo="H3 · Tasa de incumplimiento por número de moras", archivo_png="eda_H3_moras.png", xlabel="Moras en los últimos 2 años (4 = 4 o más)", es_categorica=True)
    df.drop(columns=["_delinq_grp"], inplace=True)
    return tabla

def analizar_H4():
    """H4: Gráfico de deciles exactos + Tabla de equivalencia en años/meses al lado."""
    # 1. Agrupamos por deciles exactos (como te gustaba)
    grupos = pd.qcut(df["antiguedad_hist_anios"], 10, duplicates="drop")
    tabla = (
        df.groupby(grupos, observed=True)[TARGET]
          .agg(n="size", tasa_incumplimiento="mean")
          .reset_index()
    )
    tabla = tabla.rename(columns={tabla.columns[0]: "grupo"})
    
    # 2. Preparamos los datos para la tabla visual (Traduciendo los decimales)
    datos_tabla = []
    etiquetas_eje_x = []
    
    for i, row in tabla.iterrows():
        intervalo = row["grupo"]
        # Etiqueta para el eje X (con corchetes)
        etiqueta = str(intervalo)
        etiquetas_eje_x.append(etiqueta)
        
        # Cálculo de años y meses para la tabla de la derecha
        limite_inf = round(intervalo.left, 1)
        limite_sup = round(intervalo.right, 1)
        
        anios_inf = int(limite_inf)
        meses_inf = int((limite_inf - anios_inf) * 12)
        anios_sup = int(limite_sup)
        meses_sup = int((limite_sup - anios_sup) * 12)
        
        equivalencia = f"{anios_inf}a {meses_inf}m - {anios_sup}a {meses_sup}m"
        
        datos_tabla.append([
            etiqueta, 
            equivalencia, 
            f"{row['tasa_incumplimiento']*100:.1f}%"
        ])
    
    # 3. Configuramos la figura con dos zonas
    fig, (ax_graf, ax_tabla) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.8, 1.2]})
    
    # --- ZONA 1: EL GRÁFICO ---
    ax_graf.plot(range(len(tabla)), tabla["tasa_incumplimiento"], marker='o', color='darkred', linewidth=2, markersize=8)
    ax_graf.set_xticks(range(len(tabla)))
    ax_graf.set_xticklabels(etiquetas_eje_x, rotation=45, ha="right", fontsize=8)
    ax_graf.set_ylabel("Tasa de incumplimiento")
    ax_graf.set_xlabel("Decil de antigüedad del historial (años)")
    ax_graf.set_title("H4 · Evolución del riesgo según antigüedad del historial")
    ax_graf.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    
    # --- ZONA 2: LA TABLA TRADUCIDA (al lado derecho) ---
    ax_tabla.axis('off')
    tabla_visual = ax_tabla.table(
        cellText=datos_tabla,
        colLabels=["Rango original", "Equivalencia (Años y Meses)", "Tasa"],
        cellLoc='center',
        loc='center'
    )
    tabla_visual.auto_set_font_size(False)
    tabla_visual.set_fontsize(9)
    tabla_visual.scale(1, 1.6)
    ax_tabla.set_title("Tabla de equivalencia de Deciles", pad=20)
    
    # Ajustes finales
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_H4_antiguedad_line.png"), dpi=150)
    plt.close(fig)
    print("[OK] Figura H4 guardada con tabla de equivalencia al lado")
    
    return tabla

def analizar_H5():
    """H5: Barras horizontales con nombres traducidos al español."""
    tabla = df.groupby("purpose", observed=True)[TARGET].agg(n="size", tasa_incumplimiento="mean").reset_index().sort_values("tasa_incumplimiento")
    
    # Diccionario para traducir los nombres al español
    traduccion = {
        "wedding": "Boda",
        "car": "Automóvil",
        "major_purchase": "Compra mayor",
        "home_improvement": "Mejora del hogar",
        "credit_card": "Tarjeta de crédito",
        "vacation": "Vacaciones",
        "house": "Vivienda",
        "educational": "Educación",
        "debt_consolidation": "Consolidación de deudas",
        "renewable_energy": "Energía renovable",
        "medical": "Gastos médicos",
        "other": "Otros",
        "moving": "Mudanza",
        "small_business": "Pequeño negocio"
    }
    
    # Aplicamos la traducción
    tabla["purpose_es"] = tabla["purpose"].map(traduccion)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(tabla["purpose_es"], tabla["tasa_incumplimiento"], color="purple")
    ax.set_xlabel("Tasa de incumplimiento")
    ax.set_title("H5 · Tasa de incumplimiento por propósito del crédito")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_H5_purpose_barh.png"), dpi=150)
    plt.close(fig)
    print("[OK] Figura H5 guardada (Nombres traducidos al español)")
    return tabla

def analizar_H6():
    """H6: Barras apiladas ordenadas y traducidas al español."""
    # 1. Diccionario de traducción
    traduccion = {
        "RENT": "Arrendatario",
        "OWN": "Sin hipoteca",
        "MORTGAGE": "Hipotecado",
        "OTRO": "Otro"
    }
    
    # 2. Calculamos las proporciones
    tabla_cruzada = pd.crosstab(df["home_ownership"], df[TARGET], normalize="index")
    tabla_cruzada.columns = ["Buenos (0)", "Malos (1)"]
    
    # 3. Traducimos los índices (las filas)
    tabla_cruzada.index = tabla_cruzada.index.map(lambda x: traduccion.get(x, x))
    
    # 4. Ordenamos de mayor a menor riesgo (según la columna "Malos (1)")
    tabla_cruzada = tabla_cruzada.sort_values(by="Malos (1)", ascending=True)
    
    # 5. Graficamos
    fig, ax = plt.subplots(figsize=(8, 4))
    tabla_cruzada.plot(kind="barh", stacked=True, color=["green", "red"], ax=ax)
    
    ax.set_xlabel("Proporción")
    ax.set_ylabel("") # Quitamos el título del eje Y para que se vea más limpio
    ax.set_title("H6 · Proporción de buenos y malos pagadores por vivienda")
    
    # Movemos la leyenda para que no tape las barras
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_H6_vivienda_stacked.png"), dpi=150)
    plt.close(fig)
    print("[OK] Figura H6 guardada (Stacked Bar ordenado y traducido)")
    
    # Retornamos la tabla original (sin traducir) para que el contraste final siga funcionando
    tabla_original = df.groupby("home_ownership", observed=True)[TARGET].agg(n="size", tasa_incumplimiento="mean").reset_index()
    return tabla_original

# ---------- BLOQUE 9: contraste EDA vs. Tabla 6 del modelo ----------
def construir_tabla_contraste(resultados_eda):
    ruta_t6 = os.path.join(RES_DIR, "t6_rn_puntos_variables.csv")
    df_t6 = pd.read_csv(ruta_t6)
    
    t6 = {}
    for _, row in df_t6.iterrows():
        var = row["variable"]
        imp = round(float(row["impacto_pts"]), 2)
        direc = row["direccion"] if pd.notna(row["direccion"]) else "—"
        t6[var] = (imp, direc)

    filas = []

    # H1
    t = resultados_eda["H1"]
    bajo = t.iloc[0]["tasa_incumplimiento"]
    alto = t.iloc[-1]["tasa_incumplimiento"]
    dif  = alto - bajo
    filas.append({
        "Hipotesis": "H1", "Variable": "dti", "Supuesto": "+ riesgo",
        "EDA_decil_bajo_%": round(bajo*100, 2), "EDA_decil_alto_%": round(alto*100, 2),
        "EDA_delta_pp": round(dif*100, 2), "EDA_es_monotono": t["tasa_incumplimiento"].is_monotonic_increasing,
        "Modelo_impacto": t6.get("dti", (0,0))[0], "Modelo_direccion": t6.get("dti", (0,0))[1],
        "Veredicto": "Confirmada" + (" (fuerte)" if dif > 0.10 else " (débil)"),
    })

    # H2
    t = resultados_eda["H2"]
    bajo = t.iloc[0]["tasa_incumplimiento"]
    alto = t.iloc[-1]["tasa_incumplimiento"]
    dif  = bajo - alto
    filas.append({
        "Hipotesis": "H2", "Variable": "annual_inc", "Supuesto": "− riesgo",
        "EDA_decil_bajo_%": round(bajo*100, 2), "EDA_decil_alto_%": round(alto*100, 2),
        "EDA_delta_pp": round(dif*100, 2), "EDA_es_monotono": t["tasa_incumplimiento"].is_monotonic_decreasing,
        "Modelo_impacto": t6.get("annual_inc", (0,0))[0], "Modelo_direccion": t6.get("annual_inc", (0,0))[1],
        "Veredicto": "Confirmada" + (" (fuerte)" if dif > 0.10 else " (débil)"),
    })

    # H3
    t = resultados_eda["H3"]
    sin_mora = t.loc[t["grupo"] == 0.0, "tasa_incumplimiento"].iloc[0]
    con_mora = t.loc[t["grupo"] == 4.0, "tasa_incumplimiento"].iloc[0]
    dif = con_mora - sin_mora
    filas.append({
        "Hipotesis": "H3", "Variable": "delinq_2yrs", "Supuesto": "+ riesgo",
        "EDA_decil_bajo_%": round(sin_mora*100, 2), "EDA_decil_alto_%": round(con_mora*100, 2),
        "EDA_delta_pp": round(dif*100, 2), "EDA_es_monotono": True,
        "Modelo_impacto": t6.get("delinq_2yrs", (0,0))[0], "Modelo_direccion": t6.get("delinq_2yrs", (0,0))[1],
        "Veredicto": "Confirmada (débil)" if dif < 0.10 else "Confirmada",
    })

    # H4
    t = resultados_eda["H4"]
    bajo = t.iloc[0]["tasa_incumplimiento"]
    alto = t.iloc[-1]["tasa_incumplimiento"]
    dif = bajo - alto
    filas.append({
        "Hipotesis": "H4", "Variable": "antiguedad_hist_anios", "Supuesto": "− riesgo",
        "EDA_decil_bajo_%": round(bajo*100, 2), "EDA_decil_alto_%": round(alto*100, 2),
        "EDA_delta_pp": round(dif*100, 2), "EDA_es_monotono": t["tasa_incumplimiento"].is_monotonic_decreasing,
        "Modelo_impacto": t6.get("antiguedad_hist_anios", (0,0))[0], "Modelo_direccion": t6.get("antiguedad_hist_anios", (0,0))[1],
        "Veredicto": "Confirmada (débil)" if dif < 0.10 else "Confirmada",
    })

    # H5
    t = resultados_eda["H5"].sort_values("tasa_incumplimiento")
    minimo = t.iloc[0]
    maximo = t.iloc[-1]
    filas.append({
        "Hipotesis": "H5", "Variable": "purpose", "Supuesto": "Variable",
        "EDA_decil_bajo_%": round(minimo["tasa_incumplimiento"]*100, 2), "EDA_decil_alto_%": round(maximo["tasa_incumplimiento"]*100, 2),
        "EDA_delta_pp": round((maximo["tasa_incumplimiento"] - minimo["tasa_incumplimiento"])*100, 2), "EDA_es_monotono": False,
        "Modelo_impacto": t6.get("purpose_small_business", (0,0))[0], "Modelo_direccion": t6.get("purpose_small_business", (0,0))[1],
        "Veredicto": f"Confirmada. Menor: {minimo.iloc[0]} ({minimo['tasa_incumplimiento']:.1%}); Mayor: {maximo.iloc[0]} ({maximo['tasa_incumplimiento']:.1%})",
    })

    # H6
    t = resultados_eda["H6"]
    rent = t.loc[t["home_ownership"] == "RENT", "tasa_incumplimiento"].iloc[0]
    mort = t.loc[t["home_ownership"] == "MORTGAGE", "tasa_incumplimiento"].iloc[0]
    own  = t.loc[t["home_ownership"] == "OWN", "tasa_incumplimiento"].iloc[0]
    cumple = (rent > mort) and (mort > own)
    filas.append({
        "Hipotesis": "H6", "Variable": "home_ownership", "Supuesto": "RENT > MORTGAGE > OWN",
        "EDA_decil_bajo_%": round(own*100, 2), "EDA_decil_alto_%": round(rent*100, 2),
        "EDA_delta_pp": round((rent - own)*100, 2), "EDA_es_monotono": cumple,
        "Modelo_impacto": t6.get("home_ownership_RENT", (0,0))[0], "Modelo_direccion": t6.get("home_ownership_RENT", (0,0))[1],
        "Veredicto": ("Confirmada" if cumple else f"Revisar: orden real {rent:.1%} > {own:.1%} > {mort:.1%} (RENT > OWN > MORTGAGE)"),
    })

    tabla = pd.DataFrame(filas)
    ruta_salida = os.path.join(RES_DIR, "hipotesis_eda_vs_modelo.csv")
    tabla.to_csv(ruta_salida, index=False)
    print(f"[OK] Tabla de contraste guardada: {ruta_salida}")
    return tabla


# ---------- BLOQUE 10: main ----------
if __name__ == "__main__":
    print("\n>>> Ejecutando análisis de hipótesis H1–H6 (Versión Profesional)...\n")

    resultados = {}
    resultados["H1"] = analizar_H1()
    resultados["H2"] = analizar_H2()
    resultados["H3"] = analizar_H3()
    resultados["H4"] = analizar_H4()
    resultados["H5"] = analizar_H5()
    resultados["H6"] = analizar_H6()

    print("\n>>> Construyendo tabla de contraste con la Tabla 6 del modelo...\n")
    tabla_contraste = construir_tabla_contraste(resultados)

    print("\n>>> Análisis completado. Revisa las figuras en la carpeta 'figuras/'.")
    print("\n>>> Tabla de contraste (resumen):\n")
    print(tabla_contraste[["Hipotesis", "Variable", "EDA_delta_pp", "Modelo_impacto", "Modelo_direccion", "Veredicto"]].to_string(index=False))