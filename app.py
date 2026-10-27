"""
Aplicación web de la scorecard: puntaje, probabilidad de incumplimiento (PD), posición en la población y simulador.

    streamlit run app.py

Solo carga modelos ya entrenados (Parte_1/modelos/); no reentrena ni necesita internet.
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "Parte_1"))   # el .joblib guarda un Preprocesador de Parte_1/preprocesamiento.py
from Parte_1.red_neuronal import cargar_red  
from Parte_1.scorecard_red_neuronal import logits, puntos_por_variable   

REPORTE_URL = "https://anarragan.github.io/RNA-Informe-blog/blog-2/index.html" 
VIDEO_URL = ""     # pegar aquí el enlace del video promocional

PROPOSITOS = {"debt_consolidation": "Consolidar deudas", "credit_card": "Pagar tarjeta de crédito",
              "home_improvement": "Mejoras del hogar", "small_business": "Negocio propio", "car": "Vehículo",
              "major_purchase": "Compra grande", "medical": "Gastos médicos", "moving": "Mudanza",
              "vacation": "Vacaciones", "wedding": "Boda", "house": "Casa", "educational": "Educación",
              "renewable_energy": "Energía renovable", "other": "Otro"}
VIVIENDA = {"RENT": "Alquilo", "MORTGAGE": "Hipoteca", "OWN": "Propia", "OTRO": "Otro"}
VERIFICACION = {"Not Verified": "No verificado", "Source Verified": "Fuente verificada", "Verified": "Verificado"}


@st.cache_resource
def cargar():
    sc = joblib.load(RAIZ / "Parte_1" / "modelos" / "scorecard.joblib")
    red, _ = cargar_red(str(RAIZ / "Parte_1" / "modelos" / "red_neuronal.pt"))
    return sc, red


def evaluar(perfil, sc, red, con_puntos=False):
    """Puntaje = Offset - Factor * logit; PD = sigmoide(logit). Con con_puntos, reparte el puntaje por variable."""
    prep = sc["preprocesador"]
    X = prep.transform(pd.DataFrame([perfil]))[list(prep.columnas_)]
    z = float(logits(red, X)[0])
    res = {"z": z, "pd": 1 / (1 + np.exp(-z)), "puntaje": float(sc["offset"] - sc["factor"] * z)}
    if con_puntos:
        res["puntos"] = pd.Series(puntos_por_variable(red, X, sc["x_ref"], sc["factor"])[0], index=prep.columnas_)
    return res


def percentil(puntaje, sc):
    """% de los clientes históricos con un puntaje menor o igual (es decir, con más o igual riesgo que esta persona)."""
    ref = sc["puntajes_referencia"]
    return 100 * np.searchsorted(ref, puntaje, side="right") / len(ref)


st.set_page_config(page_title="Scorecard de crédito", page_icon="💳", layout="wide")
sc, red = cargar()
st.title("¿Qué tan riesgoso es tu crédito?")
st.write(f"Una red neuronal entrenada con créditos de Lending Club (2007-2015) estima tu probabilidad de incumplimiento "
         f"y la convierte en un **puntaje**: base {sc['base']} puntos = {sc['odds']} buenos por cada malo; "
         f"cada **+{sc['pdo']} puntos duplican** esa proporción. Más puntos, menos riesgo.")
with st.sidebar:
    st.header("Enlaces")
    for nombre, url in (("Reporte técnico", REPORTE_URL), ("Video promocional", VIDEO_URL)):
        st.link_button(nombre, url) if url else st.caption(f"{nombre}: pendiente")

# 1. Formulario -------------------------------------------------------------------------------------------
with st.form("solicitud"):
    st.subheader("1. Tu solicitud")
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amnt = st.number_input("Monto solicitado (USD)", 500, 40000, 10000, 500)
        term_60m = int(st.radio("Plazo", [36, 60], horizontal=True, format_func=lambda m: f"{m} meses") == 60)
        purpose = st.selectbox("Propósito", list(PROPOSITOS), index=list(PROPOSITOS).index("debt_consolidation"),
                               format_func=PROPOSITOS.get)
        annual_inc = st.number_input("Ingreso anual (USD)", 4000, 500000, 60000, 1000)
        verification_status = st.selectbox("Verificación del ingreso", list(VERIFICACION), format_func=VERIFICACION.get)
    with c2:
        home_ownership = st.selectbox("Vivienda", list(VIVIENDA), index=1, format_func=VIVIENDA.get)
        dti = st.slider("Deuda/ingreso (DTI, %)", 0.0, 40.0, 15.0, 0.5,
                        help="Pagos mensuales de deuda sobre el ingreso mensual, sin contar este crédito")
        emp_sin = st.checkbox("No informo antigüedad laboral")
        emp_length_anios = np.nan if emp_sin else st.slider("Antigüedad laboral (años)", 0, 10, 5)
        antiguedad_hist_anios = st.slider("Antigüedad del historial crediticio (años)", 0.0, 50.0, 12.0, 0.5)
        total_acc = st.slider("Cuentas de crédito en total", 1, 80, 21)
    with c3:
        open_acc = st.slider("Cuentas abiertas", 0, 40, 9)
        revol_bal = st.number_input("Saldo rotativo (tarjetas, USD)", 0, 200000, 9500, 500)
        revol_util = st.slider("Uso del crédito rotativo (%)", 0.0, 120.0, 56.0, 1.0)
        inq_last_6mths = st.slider("Consultas al buró (últimos 6 meses)", 0, 8, 1)
        delinq_2yrs = st.slider("Moras de 30+ días (últimos 2 años)", 0, 10, 0)
        ind_mora_previa = int(st.checkbox("He tenido alguna mora en mi historial"))
        pub_rec = st.slider("Registros públicos negativos", 0, 5, 0)
        ind_registro_publico = int(pub_rec > 0)
    calcular = st.form_submit_button("Calcular", type="primary")

if calcular:
    st.session_state["perfil"] = dict(
        loan_amnt=loan_amnt, term_60m=term_60m, purpose=purpose, annual_inc=annual_inc,
        verification_status=verification_status, home_ownership=home_ownership, dti=dti,
        emp_length_anios=emp_length_anios, antiguedad_hist_anios=antiguedad_hist_anios, total_acc=total_acc,
        open_acc=open_acc, revol_bal=revol_bal, revol_util=revol_util, inq_last_6mths=inq_last_6mths,
        delinq_2yrs=delinq_2yrs, ind_mora_previa=ind_mora_previa, pub_rec=pub_rec,
        ind_registro_publico=ind_registro_publico)
perfil = st.session_state.get("perfil")
if perfil is None:
    st.info("Completa el formulario y presiona **Calcular**.")
    st.stop()

# 2. Resultado --------------------------------------------------------------------------------------------
r = evaluar(perfil, sc, red, con_puntos=True)
pct = percentil(r["puntaje"], sc)
st.subheader("2. Tu resultado")
m1, m2, m3 = st.columns(3)
m1.metric("Puntaje", f"{r['puntaje']:.0f}")
m2.metric("Probabilidad de incumplimiento (PD)", f"{100 * r['pd']:.1f} %")
m3.metric("Posición en la población", f"percentil {pct:.0f}",
          help="Porcentaje de clientes históricos con un puntaje igual o menor (más riesgo) que el tuyo")
st.write(f"El **{pct:.0f} %** de los clientes históricos tiene un puntaje igual o menor al tuyo (más riesgo); "
         f"el **{100 - pct:.0f} %** tiene un puntaje mayor.")

hist, bordes = np.histogram(sc["puntajes_referencia"], bins=40)
datos = pd.DataFrame({"puntaje": bordes[:-1], "clientes": hist})
barras = alt.Chart(datos).mark_bar(opacity=0.6).encode(x=alt.X("puntaje:Q", bin=False, title="Puntaje"),
                                                      y=alt.Y("clientes:Q", title="Clientes históricos"))
linea = alt.Chart(pd.DataFrame({"x": [r["puntaje"]]})).mark_rule(color="red", size=3).encode(x="x:Q")
st.altair_chart(barras + linea, width="stretch")

p = r["puntos"]
top = p.reindex(p.abs().sort_values(ascending=False).index[:8]).sort_values()
st.caption(f"Qué mueve tu puntaje frente al cliente promedio ({sc['puntos_base']:.0f} puntos). "
           "Positivo = suma puntos; negativo = resta.")
st.bar_chart(pd.DataFrame({"puntos": top}), horizontal=True)

# 3. Simulador --------------------------------------------------------------------------------------------
st.subheader("3. Simulador: ¿y si cambio algo?")
st.caption("Mueve los controles; el puntaje se recalcula al instante.")
s1, s2, s3, s4 = st.columns(4)
f_dti = s1.slider("Deuda/ingreso (DTI)", -50, 50, 0, 5, format="%d %%")
f_inc = s2.slider("Ingreso anual", -50, 50, 0, 5, format="%d %%")
f_monto = s3.slider("Monto solicitado", -50, 50, 0, 5, format="%d %%")
f_util = s4.slider("Uso del crédito rotativo", -50, 50, 0, 5, format="%d %%")
plazo = st.radio("Plazo", ["igual", 36, 60], horizontal=True,
                 format_func=lambda v: "El mismo" if v == "igual" else f"{v} meses")
nuevo = dict(perfil, dti=perfil["dti"] * (1 + f_dti / 100), annual_inc=perfil["annual_inc"] * (1 + f_inc / 100),
             loan_amnt=perfil["loan_amnt"] * (1 + f_monto / 100), revol_util=perfil["revol_util"] * (1 + f_util / 100))
if plazo != "igual":
    nuevo["term_60m"] = int(plazo == 60)
rn = evaluar(nuevo, sc, red)
n1, n2, n3 = st.columns(3)
n1.metric("Puntaje simulado", f"{rn['puntaje']:.0f}", f"{rn['puntaje'] - r['puntaje']:+.1f} puntos")
n2.metric("PD simulada", f"{100 * rn['pd']:.1f} %", f"{100 * (rn['pd'] - r['pd']):+.1f} pp", delta_color="inverse")
n3.metric("Percentil simulado", f"{percentil(rn['puntaje'], sc):.0f}")
st.caption("Modelo educativo entrenado con datos históricos de Lending Club; no es una oferta ni una decisión de crédito.")
