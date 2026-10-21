"""
Preprocesamiento compartido por la regresión logística y, más adelante, la red neuronal.

Todo lo que se calcula con datos (medianas, límites de recorte, medias,
desviaciones) se ajusta SOLO con entrenamiento y después se aplica igual a validación y prueba.
Así los tres conjuntos se tratan igual y no se filtra información del futuro.
"""
import numpy as np
import pandas as pd

RUTA_DATOS = "data/datos_modelo.csv"

# Variables numéricas continuas y conteos: se imputan, se recortan y se estandarizan
CONTINUAS = ["loan_amnt", "annual_inc", "dti", "revol_bal", "revol_util",
             "antiguedad_hist_anios", "emp_length_anios"]
CONTEOS = ["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc"]

# Ya son 0/1: se dejan como están (así su coeficiente se lee directo como razón de odds)
BINARIAS = ["term_60m", "ind_mora_previa", "ind_registro_publico"]

# Categóricas -> variables 0/1 (one-hot). Se indica la categoría de referencia, que se omite.
CATEGORICAS = {
    "purpose": "debt_consolidation",
    "home_ownership": "MORTGAGE",
    "verification_status": "Not Verified",
}

# Variables muy sesgadas a la derecha: se usa log(1 + x) antes de estandarizar
CON_LOG = ["annual_inc", "revol_bal"]

def cargar_datos(ruta=RUTA_DATOS):
    return pd.read_csv(ruta, parse_dates=["issue_d"])

def partir(df, corte_val, corte_test):
    """Partición temporal por fecha de originación: entrenamiento < validación < prueba."""
    corte_val, corte_test = pd.Timestamp(corte_val), pd.Timestamp(corte_test)
    if not corte_val < corte_test:
        raise ValueError("El corte de validación debe ser anterior al corte de prueba.")
    train = df[df["issue_d"] < corte_val]
    val = df[(df["issue_d"] >= corte_val) & (df["issue_d"] < corte_test)]
    test = df[df["issue_d"] >= corte_test]
    for nombre, parte in [("entrenamiento", train), ("validación", val), ("prueba", test)]:
        if parte.empty:
            raise ValueError(f"La partición de {nombre} quedó vacía; revisa los cortes.")
    return train.copy(), val.copy(), test.copy()

class Preprocesador:
    """Imputa con la mediana, recorta extremos por percentiles, estandariza y codifica categorías."""

    def __init__(self, p_inf=1, p_sup=99):
        self.p_inf = p_inf
        self.p_sup = p_sup

    def _numericas(self, df):
        X = df[CONTINUAS + CONTEOS].astype(float).copy()
        for c in CON_LOG:
            X[c] = np.log1p(X[c])
        return X

    def fit(self, df):
        X = self._numericas(df)
        self.medianas_ = X.median()
        X = X.fillna(self.medianas_)
        self.lim_inf_ = X.quantile(self.p_inf / 100)
        self.lim_sup_ = X.quantile(self.p_sup / 100)
        X = X.clip(self.lim_inf_, self.lim_sup_, axis=1)
        self.medias_ = X.mean()
        self.desv_ = X.std(ddof=0).replace(0, 1.0)  # evita dividir por cero
        self.categorias_ = {}
        for var, ref in CATEGORICAS.items():
            cats = sorted(df[var].dropna().unique())
            if ref not in cats:
                raise ValueError(f"La categoría de referencia '{ref}' no está en '{var}' (entrenamiento).")
            self.categorias_[var] = cats
        self.columnas_ = list(self.transform(df).columns)
        return self

    def transform(self, df):
        X = self._numericas(df)
        emp_nulo = X["emp_length_anios"].isna().astype(int)  # se marca ANTES de imputar
        X = X.fillna(self.medianas_)
        X = X.clip(self.lim_inf_, self.lim_sup_, axis=1)
        X = (X - self.medias_) / self.desv_
        X["emp_length_nulo"] = emp_nulo
        for c in BINARIAS:
            X[c] = df[c].astype(int)
        for var, ref in CATEGORICAS.items():
            for cat in self.categorias_[var]:
                if cat != ref:
                    X[f"{var}_{cat}"] = (df[var] == cat).astype(int)
        # Una categoría nunca vista en entrenamiento queda con todos los 0/1 en 0,
        # es decir, se trata como la categoría de referencia.
        return X