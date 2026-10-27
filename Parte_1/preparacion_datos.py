"""
Paso 1: de los datos crudos de Lending Club a los datos de modelamiento.

    ENTRADA:  loan.csv (Kaggle "ranadeep/credit-risk-dataset", se descarga con kagglehub)
    SALIDA 1: data/datos_modelo.csv  -> lo usan TODAS las tareas (regresión logística, red, scorecard...)
    SALIDA 2: data/base_lgd.csv      -> solo créditos malos, con los campos de recuperación (Parte 2)

Esta es la ÚNICA clase que limpia datos. Los demás archivos leen la salida 1.

"""
import argparse
import os

import pandas as pd

# Predictoras que se usan directamente; el resto de las usadas se transforman en construir_dataset.
PREDICTORAS = [
    "loan_amnt", "purpose", "home_ownership", "annual_inc", "verification_status", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
]
# Se excluyen (caso de uso: solo información disponible al solicitar, sin fuga de datos):
#   - posteriores al desembolso: out_prncp*, total_pymnt*, total_rec_*, recoveries, collection_recovery_fee,
#     last_pymnt_*, next_pymnt_d, last_credit_pull_d, pymnt_plan
#   - decisiones de la plataforma: grade, sub_grade, int_rate, installment, funded_amnt*
#   - casi vacías (>97 % nulos), constantes, identificadores/texto libre y geografía (posible proxy de variables protegidas)

# Variable objetivo: malo = 1, bueno = 0
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

class PreparadorDatos:
    """Convierte loan.csv en datos listos para modelar."""

    def __init__(self, ruta_csv=None, salida="data/datos_modelo.csv", salida_lgd="data/base_lgd.csv"):
        self.ruta_csv = ruta_csv      
        self.salida = salida
        self.salida_lgd = salida_lgd

    def cargar(self):
        """Lee loan.csv. Devuelve el DataFrame crudo con issue_d ya convertida a fecha."""
        ruta = self.ruta_csv
        if ruta is None:
            import kagglehub  # solo se necesita en este paso
            ruta = os.path.join(kagglehub.dataset_download("ranadeep/credit-risk-dataset"), "loan", "loan.csv")
        df = pd.read_csv(ruta, low_memory=False)

        print("1. DATOS CRUDOS")
        print(f"Registros y variables: {df.shape[0]:,} x {df.shape[1]}")
        # Se convierte a fecha ANTES de sacar mínimo y máximo. 
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
        print(f"Periodo (issue_d): {df['issue_d'].min():%Y-%m} a {df['issue_d'].max():%Y-%m}")
        return df

    def construir_objetivo(self, df):
        """Crea la columna 'malo' (1 = incumple) y quita los créditos sin desenlace claro."""
        df = df.copy()
        df["malo"] = df["loan_status"].map(MAPA_OBJETIVO)  
        n_excluidos = int(df["malo"].isna().sum())
        df = df.dropna(subset=["malo"]).copy()             
        df["malo"] = df["malo"].astype(int)

        print("\n2. OBJETIVO")
        print(f"Registros modelables: {len(df):,}")
        print(f"  Buenos (0): {(df['malo'] == 0).sum():,}")
        print(f"  Malos  (1): {(df['malo'] == 1).sum():,}")
        print(f"  Tasa de malos: {df['malo'].mean():.3f}")
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

        print("\n3. DATASET DE MODELAMIENTO")
        print(f"Filas x columnas: {d.shape[0]:,} x {d.shape[1]}")
        print("Columnas:", ", ".join(d.columns))
        faltantes = d.isna().sum()
        faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
        print("\nValores faltantes reales (se imputaran despues, usando solo datos de entrenamiento):")
        for col, n in faltantes.items():
            print(f"  {col}: {n:,} ({n / len(d) * 100:.2f}%)")
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
        """Corre todo el paso: cargar -> objetivo -> dataset -> guardar los dos archivos."""
        df = self.cargar()
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
