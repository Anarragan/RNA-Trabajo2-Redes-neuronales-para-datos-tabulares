"""
Datos listos para modelar: partición temporal + preprocesamiento.
Lo usan la regresión logística y, más adelante, la red neuronal, la scorecard y la parte 2,
para estandarizar los datos y que tengan el mismo tratamiento.

Uso típico:
    from preprocesamiento import preparar
    prep, X, y, partes = preparar()
    X["entrenamiento"], y["entrenamiento"]   # y lo mismo con "validacion" y "prueba"
"""
import numpy as np
import pandas as pd

RUTA_DATOS = "data/datos_modelo.csv"

# Cortes de la partición temporal (por fecha de originación). Cambiarlos aquí los cambia en todos los modelos.
CORTE_VAL = "2013-01-01"    # validación: desde esta fecha
CORTE_TEST = "2014-01-01"   # prueba: desde esta fecha
NOMBRES = ("entrenamiento", "validacion", "prueba")

# Continuas y conteos: se imputan con la mediana, se recortan por percentiles y se estandarizan
CONTINUAS = ["loan_amnt", "annual_inc", "dti", "revol_bal", "revol_util",
             "antiguedad_hist_anios", "emp_length_anios"]
CONTEOS = ["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc"]

# Ya son 0/1: se dejan como están
BINARIAS = ["term_60m", "ind_mora_previa", "ind_registro_publico"]

# Categóricas -> columnas 0/1 (one-hot). Se indica la categoría de referencia, que se omite.
CATEGORICAS = {
    "purpose": "debt_consolidation",
    "home_ownership": "MORTGAGE",
    "verification_status": "Not Verified",
}

# Muy sesgadas a la derecha: se usa log(1 + x) antes de estandarizar
CON_LOG = ["annual_inc", "revol_bal"]

def partir(df, corte_val=CORTE_VAL, corte_test=CORTE_TEST):
    """Partición temporal: entrenamiento < validación < prueba."""
    corte_val, corte_test = pd.Timestamp(corte_val), pd.Timestamp(corte_test)
    if not corte_val < corte_test:
        raise ValueError("El corte de validación debe ser anterior al corte de prueba.")
    partes = {
        "entrenamiento": df[df["issue_d"] < corte_val],
        "validacion": df[(df["issue_d"] >= corte_val) & (df["issue_d"] < corte_test)],
        "prueba": df[df["issue_d"] >= corte_test],
    }
    for nombre, parte in partes.items():
        if parte.empty:
            raise ValueError(f"La partición de {nombre} quedó vacía; revisa los cortes.")
    return {k: v.copy() for k, v in partes.items()}


class Preprocesador:
    """Imputa con la mediana, recorta extremos, estandariza y codifica categorías."""

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
        # Una categoría que no estaba en entrenamiento queda con todos sus 0/1 en 0,
        # es decir, se trata como la categoría de referencia.
        return X


def preparar(corte_val=CORTE_VAL, corte_test=CORTE_TEST, ruta=RUTA_DATOS):
    """Todo el flujo de datos en una llamada.

    Devuelve (preprocesador, X, y, partes); X, y y partes son diccionarios con las claves
    "entrenamiento", "validacion" y "prueba".
    """
    partes = partir(pd.read_csv(ruta, parse_dates=["issue_d"]), corte_val, corte_test)
    prep = Preprocesador().fit(partes["entrenamiento"])
    X = {k: prep.transform(v) for k, v in partes.items()}
    y = {k: v["malo"].to_numpy() for k, v in partes.items()}
    return prep, X, y, partes