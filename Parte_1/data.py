"""
Paso 1: separar las variables de Lending Club (predictoras / futuro / descartadas)
y armar el dataset de modelamiento: data/datos_modelo.csv

Ejecutar (sin argumentos):  python data.py
"""
import os

import kagglehub
import pandas as pd

SALIDA = "data/datos_modelo.csv"


def titulo(texto):
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)


# ---------------------------------------------------------------
# 1. CARGAR LOS DATOS Y VER EL PERIODO
# ---------------------------------------------------------------
path = kagglehub.dataset_download("ranadeep/credit-risk-dataset")
df = pd.read_csv(os.path.join(path, "loan", "loan.csv"), low_memory=False)

titulo("1. DATOS")
print(f"Registros y variables: {df.shape[0]:,} x {df.shape[1]}")

nulos = df.isna().mean() * 100  # % de nulos por columna, con todos los datos

# Se convierte a fecha ANTES de sacar mínimo y máximo. Como texto, el orden
# es alfabético ("Apr-2008" va antes que "Aug-2007") y el periodo sale mal.
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
print(f"Periodo (issue_d): {df['issue_d'].min():%Y-%m} a {df['issue_d'].max():%Y-%m}")

# ---------------------------------------------------------------
# 2. GRUPOS DE VARIABLES (las 74 columnas quedan en un solo grupo)
# ---------------------------------------------------------------
# Se usan tal cual
PREDICTORAS = [
    "loan_amnt", "purpose", "home_ownership", "annual_inc", "verification_status", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
]

# Se usan, pero convertidas a otra forma (ver sección 4)
A_TRANSFORMAR = [
    "term",                     # texto "36 months" / "60 months" -> indicador de 60 meses
    "emp_length",               # texto "10+ years" -> número de años
    "earliest_cr_line",         # fecha -> antigüedad del historial
    "mths_since_last_delinq",   # su nulo significa "nunca tuvo mora": se vuelve indicador
    "mths_since_last_record",   # su nulo significa "nunca tuvo registro público": indicador
    "issue_d",                  # fecha del crédito: sirve para dividir en el tiempo, no es predictora
]

OBJETIVO = ["loan_status"]

# FUTURO: solo se conocen DESPUÉS de dar el crédito. Usarlas sería hacer trampa:
# el modelo vería el resultado antes de predecirlo.
POSTERIORES = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "pymnt_plan",
]

# Decisiones de Lending Club: el solicitante no las conoce al aplicar
PLATAFORMA = ["grade", "sub_grade", "int_rate", "installment", "funded_amnt", "funded_amnt_inv"]

# Casi vacías (más de 97 % de nulos)
MUCHOS_NULOS = [
    "dti_joint", "annual_inc_joint", "verification_status_joint", "il_util", "mths_since_rcnt_il",
    "inq_fi", "open_rv_24m", "open_acc_6m", "all_util", "inq_last_12m", "total_cu_tl",
    "open_il_12m", "max_bal_bc", "open_il_6m", "open_il_24m", "open_rv_12m", "total_bal_il",
]

NO_APORTAN = ["policy_code", "application_type", "initial_list_status"]
IDENTIFICADORES_TEXTO = ["id", "member_id", "url", "desc", "title", "emp_title"]
GEOGRAFIA = ["zip_code", "addr_state"]  # posible proxy de características protegidas

# Podrían servir en una versión mejorada, pero no en el modelo básico
OTRAS = [
    "tot_cur_bal", "tot_coll_amt", "total_rev_hi_lim",
    "collections_12_mths_ex_med", "acc_now_delinq", "mths_since_last_major_derog",
]

GRUPOS = {
    "Predictoras directas (se usan)": PREDICTORAS,
    "Predictoras transformadas (se usan)": A_TRANSFORMAR,
    "Objetivo": OBJETIVO,
    "Posteriores al crédito (futuro, se excluyen)": POSTERIORES,
    "Decisiones de la plataforma (se excluyen)": PLATAFORMA,
    "Casi vacías (se excluyen)": MUCHOS_NULOS,
    "No aportan (se excluyen)": NO_APORTAN,
    "Identificadores y texto (se excluyen)": IDENTIFICADORES_TEXTO,
    "Geografía (se excluye)": GEOGRAFIA,
    "Otras (versión mejorada)": OTRAS,
}

titulo("2. CLASIFICACION DE LAS 74 COLUMNAS")
for nombre, lista in GRUPOS.items():
    print(f"{len(lista):>3}  {nombre}")

# Verificación: ninguna columna sin clasificar, ninguna en dos grupos
todas = [c for lista in GRUPOS.values() for c in lista]
sin_clasificar = [c for c in df.columns if c not in todas]
repetidas = sorted({c for c in todas if todas.count(c) > 1})
no_existen = [c for c in todas if c not in df.columns]
if not (sin_clasificar or repetidas or no_existen):
    print("Verificacion: OK, cada columna esta en un solo grupo.")
else:
    print("Verificacion: PROBLEMA")
    print("  Sin clasificar:", sin_clasificar)
    print("  En dos grupos:", repetidas)
    print("  No existen en el CSV:", no_existen)

print(f"\nCasi vacias: entre {nulos[MUCHOS_NULOS].min():.1f}% y {nulos[MUCHOS_NULOS].max():.1f}% de nulos.")
print("\nVariables 'no aportan' (valores y cuantos registros):")
for c in NO_APORTAN:
    print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")
print(f"  pymnt_plan (excluida por ser posterior): {df['pymnt_plan'].value_counts(dropna=False).to_dict()}")

# ---------------------------------------------------------------
# 3. VARIABLE OBJETIVO: malo = 1, bueno = 0, el resto se elimina
# ---------------------------------------------------------------
mapa = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}
df["malo"] = df["loan_status"].map(mapa)  # lo que no está en el mapa queda NA
n_excluidos = int(df["malo"].isna().sum())
df = df.dropna(subset=["malo"]).copy()    # quita Current, Issued, In Grace Period, Late (16-30 days)
df["malo"] = df["malo"].astype(int)

titulo("3. VARIABLE OBJETIVO")
print(f"Excluidos (Current, Issued, In Grace Period, Late 16-30): {n_excluidos:,}")
print(f"Registros modelables: {len(df):,}")
print(f"  Buenos (0): {(df['malo'] == 0).sum():,}")
print(f"  Malos  (1): {(df['malo'] == 1).sum():,}")
print(f"  Tasa de malos: {df['malo'].mean():.3f}")
print(f"Solicitudes conjuntas (JOINT) entre los modelables: {(df['application_type'] == 'JOINT').sum():,}")

# ---------------------------------------------------------------
# 4. DATASET PARA EL MODELO
# ---------------------------------------------------------------
d = df[["issue_d", "malo"] + PREDICTORAS].copy()

# plazo: 0 = 36 meses, 1 = 60 meses
d["term_60m"] = df["term"].str.strip().map({"36 months": 0, "60 months": 1})

# antigüedad laboral en años ("< 1 year" -> 0, "10+ years" -> 10, "n/a" -> vacío)
emp = df["emp_length"].str.extract(r"(\d+)")[0].astype(float)
d["emp_length_anios"] = emp.mask(df["emp_length"].fillna("").str.startswith("<"), 0)

# vivienda: agrupa las categorías raras
ho = d["home_ownership"].str.upper()
d["home_ownership"] = ho.where(ho.isin(["MORTGAGE", "OWN", "RENT"]), "OTRO")

# antigüedad del historial, medida hasta la fecha del crédito (no hasta hoy)
earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
d["antiguedad_hist_anios"] = (df["issue_d"] - earliest).dt.days / 365.25

# indicadores: 1 si la persona tuvo el evento, 0 si el dato está vacío (nunca tienen nulos)
d["ind_mora_previa"] = df["mths_since_last_delinq"].notna().astype(int)
d["ind_registro_publico"] = df["mths_since_last_record"].notna().astype(int)

titulo("4. DATASET FINAL")
print(f"Filas x columnas: {d.shape[0]:,} x {d.shape[1]}")
print("Columnas:", ", ".join(d.columns))

faltantes = d.isna().sum()
faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
print("\nValores faltantes reales (se imputaran despues, usando solo datos de entrenamiento):")
for col, n in faltantes.items():
    print(f"  {col}: {n:,} ({n / len(d) * 100:.2f}%)")
print("Las demas columnas, incluidos los indicadores ind_*, no tienen faltantes.")

os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
d.to_csv(SALIDA, index=False)
print(f"\nGuardado en {SALIDA} ({os.path.getsize(SALIDA) / 1e6:.1f} MB)")