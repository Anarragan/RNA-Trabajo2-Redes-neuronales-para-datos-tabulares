"""
Paso 1: separar las variables de Lending Club en grupos (predictoras / futuro / descartadas)
y armar el dataset que usará la regresión logística.

Ejecutar (sin argumentos):  python preparar_variables.py
"""
import os
import kagglehub
import pandas as pd

def titulo(texto):
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)

# ---------------------------------------------------------------
# 1. CARGAR LOS DATOS Y VER EL PERIODO
# ---------------------------------------------------------------
path = kagglehub.dataset_download("ranadeep/credit-risk-dataset")
csv_path = os.path.join(path, "loan", "loan.csv")
df = pd.read_csv(csv_path, low_memory=False)

titulo("1. DATOS")
print(f"Registros y variables: {df.shape[0]:,} x {df.shape[1]}")

# Porcentaje de nulos de cada columna (con todos los datos, antes de filtrar)
nulos = df.isna().mean() * 100

# Se convierte a fecha ANTES de sacar mínimo y máximo. Como texto, el orden
# es alfabético ("Apr-2008" va antes que "Aug-2007") y el periodo sale mal.
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
print(f"Periodo (issue_d): {df['issue_d'].min():%Y-%m} a {df['issue_d'].max():%Y-%m}")

# ---------------------------------------------------------------
# 2. GRUPOS DE VARIABLES )
# ---------------------------------------------------------------
# Las que SÍ usamos tal cual
PREDICTORAS = [
    "loan_amnt", "term", "purpose", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
]

# Las que se usan, pero transformadas
A_TRANSFORMAR = [
    "earliest_cr_line",         # se convierte en antiguedad_hist_anios
    "mths_since_last_delinq",   # su nulo significa "nunca tuvo mora": se vuelve  ind_mora_previa
    "mths_since_last_record",   # su nulo significa "nunca tuvo registro público":  ind_registro_publico
    "issue_d",                  # fecha del crédito: sirve para dividir en el tiempo, no es predictora
]

OBJETIVO = ["loan_status"]

# FUTURO: solo se conocen DESPUÉS de dar el crédito (pagos, saldos, recuperaciones...)
# Usarlas sería hacer trampa: el modelo vería el resultado antes de predecirlo.
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

# No aportan: constantes o sin relación con el solicitante
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
df["malo"] = df["loan_status"].map(mapa)
n_excluidos = int(df["malo"].isna().sum())
df = df.dropna(subset=["malo"]).copy()  # quita Current, Issued, In Grace Period, Late (16-30 days)
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
df_modelo = df[["issue_d", "malo"] + PREDICTORAS].copy()

# plazo: 0 = 36 meses, 1 = 60 meses
df_modelo["term_60m"] = df_modelo.pop("term").str.strip().map({"36 months": 0, "60 months": 1})

# antigüedad laboral en años ("< 1 year" -> 0, "10+ years" -> 10, "n/a" -> vacío)
emp = df_modelo["emp_length"].str.extract(r"(\d+)")[0].astype(float)
emp = emp.mask(df_modelo["emp_length"].fillna("").str.startswith("<"), 0)
df_modelo["emp_length_anios"] = emp
df_modelo = df_modelo.drop(columns="emp_length")

# vivienda: agrupa las categorías raras
ho = df_modelo["home_ownership"].str.upper()
df_modelo["home_ownership"] = ho.where(ho.isin(["MORTGAGE", "OWN", "RENT"]), "OTRO")

# antigüedad del historial, medida hasta la fecha del crédito (no hasta hoy)
earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
df_modelo["antiguedad_hist_anios"] = (df["issue_d"] - earliest).dt.days / 365.25

# indicadores: 1 si la persona tuvo el evento, 0 si el dato está vacío
# (por construcción nunca tienen nulos)
df_modelo["ind_mora_previa"] = df["mths_since_last_delinq"].notna().astype(int)
df_modelo["ind_registro_publico"] = df["mths_since_last_record"].notna().astype(int)

titulo("4. DATASET FINAL")
print(f"Filas x columnas: {df_modelo.shape[0]:,} x {df_modelo.shape[1]}")
print("Columnas:", ", ".join(df_modelo.columns))

faltantes = df_modelo.isna().sum()
faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
print("\nValores faltantes reales (se imputaran despues, usando solo datos de entrenamiento):")
for col, n in faltantes.items():
    print(f"  {col}: {n:,} ({n / len(df_modelo) * 100:.2f}%)")
print("Las demas columnas, incluidos los indicadores ind_*, no tienen faltantes.")

os.makedirs("data", exist_ok=True)
salida = "data/datos_modelo.csv"
df_modelo.to_csv(salida, index=False)
print(f"\nGuardado en {salida} ({os.path.getsize(salida) / 1e6:.1f} MB)")