"""
Paso 1: de los datos crudos de Lending Club a los datos de modelamiento.

    ENTRADA:  loan.csv (Kaggle "ranadeep/credit-risk-dataset", se descarga con kagglehub)
    SALIDA 1: data/datos_modelo.csv  -> lo usan TODAS las tareas (regresión logística, red, scorecard...)
    SALIDA 2: data/base_lgd.csv      -> solo créditos malos, con los campos de recuperación (Parte 2)

Esta es la ÚNICA clase que limpia datos. Los demás archivos leen la salida 1.

Ejecutar:
    python preparacion_datos.py                     # descarga el dataset con kagglehub
    python preparacion_datos.py --csv ruta/loan.csv  # usa un loan.csv que ya tengas
"""
import argparse
import os

import pandas as pd

# ---------------------------------------------------------------
# GRUPOS DE VARIABLES (las 74 columnas quedan en un solo grupo)
# ---------------------------------------------------------------
# Se usan tal cual
PREDICTORAS = [
    "loan_amnt", "purpose", "home_ownership", "annual_inc", "verification_status", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
]

# Se usan, pero convertidas a otra forma (ver construir_dataset)
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

# Variable objetivo: malo = 1, bueno = 0; lo que no está aquí (Current, Issued, ...) queda NA y se elimina
MAPA_OBJETIVO = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Parte 2 (LGD): campos de recuperación de los créditos malos. Solo se guardan, no se interpretan aquí.
CAMPOS_LGD = [
    "id", "loan_status", "funded_amnt", "out_prncp", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
]


def titulo(texto):
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)


class PreparadorDatos:
    """Convierte loan.csv (crudo) en data/datos_modelo.csv y data/base_lgd.csv (listos para modelar)."""

    def __init__(self, ruta_csv=None, salida="data/datos_modelo.csv", salida_lgd="data/base_lgd.csv"):
        self.ruta_csv = ruta_csv      # None = descargar con kagglehub
        self.salida = salida
        self.salida_lgd = salida_lgd

    def cargar(self):
        """Lee loan.csv. Devuelve el DataFrame crudo con issue_d ya convertida a fecha."""
        ruta = self.ruta_csv
        if ruta is None:
            import kagglehub  # solo se necesita en este paso
            ruta = os.path.join(kagglehub.dataset_download("ranadeep/credit-risk-dataset"), "loan", "loan.csv")
        df = pd.read_csv(ruta, low_memory=False)

        titulo("1. DATOS")
        print(f"Registros y variables: {df.shape[0]:,} x {df.shape[1]}")
        # Se convierte a fecha ANTES de sacar mínimo y máximo. Como texto, el orden es
        # alfabético ("Apr-2008" va antes que "Aug-2007") y el periodo sale mal.
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
        print(f"Periodo (issue_d): {df['issue_d'].min():%Y-%m} a {df['issue_d'].max():%Y-%m}")
        return df

    def revisar_columnas(self, df):
        """Imprime cuántas columnas hay en cada grupo y verifica que ninguna quede sin clasificar."""
        titulo("2. CLASIFICACION DE LAS 74 COLUMNAS")
        for nombre, lista in GRUPOS.items():
            print(f"{len(lista):>3}  {nombre}")

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

        nulos = df[MUCHOS_NULOS].isna().mean() * 100
        print(f"\nCasi vacias: entre {nulos.min():.1f}% y {nulos.max():.1f}% de nulos.")
        print("\nVariables 'no aportan' (valores y cuantos registros):")
        for c in NO_APORTAN:
            print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")
        print(f"  pymnt_plan (excluida por ser posterior): {df['pymnt_plan'].value_counts(dropna=False).to_dict()}")

    def construir_objetivo(self, df):
        """Crea la columna 'malo' (1 = incumple) y quita los créditos sin desenlace claro."""
        df = df.copy()
        df["malo"] = df["loan_status"].map(MAPA_OBJETIVO)  # lo que no está en el mapa queda NA
        n_excluidos = int(df["malo"].isna().sum())
        df = df.dropna(subset=["malo"]).copy()             # quita Current, Issued, In Grace Period, Late (16-30 days)
        df["malo"] = df["malo"].astype(int)

        titulo("3. VARIABLE OBJETIVO")
        print(f"Excluidos (Current, Issued, In Grace Period, Late 16-30): {n_excluidos:,}")
        print(f"Registros modelables: {len(df):,}")
        print(f"  Buenos (0): {(df['malo'] == 0).sum():,}")
        print(f"  Malos  (1): {(df['malo'] == 1).sum():,}")
        print(f"  Tasa de malos: {df['malo'].mean():.3f}")
        print(f"Solicitudes conjuntas (JOINT) entre los modelables: {(df['application_type'] == 'JOINT').sum():,}")
        return df

    def construir_dataset(self, df):
        """Deja solo las predictoras (ya transformadas), la fecha y 'malo'. Devuelve el DataFrame del modelo."""
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
        return d

    def construir_base_lgd(self, df, d):
        """Parte 2: créditos malos con sus campos de recuperación, alineados con las columnas del modelo."""
        malos = d["malo"] == 1
        return pd.concat([d[malos], df.loc[malos.index[malos], CAMPOS_LGD]], axis=1)

    def guardar(self, tabla, ruta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        tabla.to_csv(ruta, index=False)
        print(f"Guardado en {ruta} ({os.path.getsize(ruta) / 1e6:.1f} MB)")

    def ejecutar(self):
        """Corre todo el paso: cargar -> revisar -> objetivo -> dataset -> guardar los dos archivos."""
        df = self.cargar()
        self.revisar_columnas(df)
        df = self.construir_objetivo(df)
        d = self.construir_dataset(df)
        print()
        self.guardar(d, self.salida)
        self.guardar(self.construir_base_lgd(df, d), self.salida_lgd)
        return d


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Paso 1: preparar los datos de modelamiento")
    parser.add_argument("--csv", default=None, help="ruta de un loan.csv local (si no, se descarga con kagglehub)")
    args = parser.parse_args()
    PreparadorDatos(ruta_csv=args.csv).ejecutar()
