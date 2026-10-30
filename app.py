"""
Aplicación web integrada de Riesgo de Crédito:
- Clasificación: Probabilidad de Incumplimiento (PD) y Scorecard de Crédito.
- Regresión: Pérdida Dado el Incumplimiento (LGD).
- Métricas Financieras: Exposición (EAD) y Pérdida Esperada E[L] = PD * LGD * EAD.
- Diagnóstico visual y simulador "What-If" en tiempo real.

Uso:
    streamlit run app.py
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
if str(RAIZ / "Parte_1") not in sys.path:
    sys.path.insert(0, str(RAIZ / "Parte_1"))
if str(RAIZ / "Parte_2") not in sys.path:
    sys.path.insert(0, str(RAIZ / "Parte_2"))

from Parte_1.red_neuronal import cargar_red as cargar_red_pd
from Parte_1.scorecard_red_neuronal import logits, puntos_por_variable
from Parte_2.red_neuronal_lgd import cargar_red_lgd, predecir_lgd

REPORTE_URL = "https://anarragan.github.io/RNA-Informe-blog/blog-2/index.html"
VIDEO_URL = ""  # Enlace al video promocional una vez publicado

PROPOSITOS = {
    "debt_consolidation": "Consolidar deudas",
    "credit_card": "Pagar tarjeta de crédito",
    "home_improvement": "Mejoras del hogar",
    "small_business": "Negocio propio",
    "car": "Vehículo",
    "major_purchase": "Compra grande",
    "medical": "Gastos médicos",
    "moving": "Mudanza",
    "vacation": "Vacaciones",
    "wedding": "Boda",
    "house": "Casa",
    "educational": "Educación",
    "renewable_energy": "Energía renovable",
    "other": "Otro",
}
VIVIENDA = {"RENT": "Alquilo", "MORTGAGE": "Hipoteca", "OWN": "Propia", "OTRO": "Otro"}
VERIFICACION = {
    "Not Verified": "No verificado",
    "Source Verified": "Fuente verificada",
    "Verified": "Verificado",
}


@st.cache_resource
def cargar():
    """Carga los modelos entrenados de PD (Parte 1) y LGD (Parte 2)."""
    sc = joblib.load(RAIZ / "Parte_1" / "modelos" / "scorecard.joblib")
    red_pd, _ = cargar_red_pd(str(RAIZ / "Parte_1" / "modelos" / "red_neuronal.pt"))
    prep_lgd = joblib.load(RAIZ / "Parte_2" / "modelos" / "preprocesador_lgd.joblib")
    red_lgd, _ = cargar_red_lgd(str(RAIZ / "Parte_2" / "modelos" / "red_neuronal_lgd.pt"))
    return sc, red_pd, prep_lgd, red_lgd


def agregar_variables_derivadas(p):
    """Calcula las variables ingenieriles ex-ante a partir de las entradas básicas."""
    p = dict(p)
    p["ratio_prestamo_ingreso"] = p["loan_amnt"] / (p["annual_inc"] + 1)
    p["ingreso_libre"] = p["annual_inc"] * (1 - (p["dti"] / 100))
    p["saldo_promedio_cuenta"] = p["revol_bal"] / (p["open_acc"] + 1)
    p["ratio_cuentas_activas"] = p["open_acc"] / (p["total_acc"] + 1)
    return p


def evaluar(perfil, sc, red_pd, prep_lgd, red_lgd, con_puntos=False):
    """
    Evalúa integralmente el perfil del cliente:
    - PD y Scorecard mediante la red neuronal de clasificación y Gradientes Integrados.
    - LGD mediante la red neuronal de regresión acotada [0, 1].
    - Pérdida Esperada E[L] = PD * LGD * EAD.
    """
    p_completo = agregar_variables_derivadas(perfil)
    df_p = pd.DataFrame([p_completo])

    # 1. Probabilidad de Incumplimiento (PD) y Scorecard
    prep_pd = sc["preprocesador"]
    X_pd = prep_pd.transform(df_p)[list(prep_pd.columnas_)]
    z = float(logits(red_pd, X_pd)[0])
    pd_val = float(1 / (1 + np.exp(-z)))
    puntaje = float(sc["offset"] - sc["factor"] * z)

    # 2. Pérdida Dado el Incumplimiento (LGD)
    X_lgd = prep_lgd.transform(df_p)[list(prep_lgd.columnas_)]
    lgd_val = float(predecir_lgd(red_lgd, X_lgd)[0])

    # 3. Exposición al Incumplimiento (EAD) y Pérdida Esperada E[L]
    ead = float(perfil["loan_amnt"])
    el = float(pd_val * lgd_val * ead)

    res = {
        "z": z,
        "pd": pd_val,
        "puntaje": puntaje,
        "lgd": lgd_val,
        "ead": ead,
        "el": el,
    }
    if con_puntos:
        res["puntos"] = pd.Series(
            puntos_por_variable(red_pd, X_pd, sc["x_ref"], sc["factor"])[0],
            index=prep_pd.columnas_,
        )
    return res


def percentil(puntaje, sc):
    """Calcula el percentil histórico frente a la población de Lending Club."""
    ref = sc["puntajes_referencia"]
    return 100 * np.searchsorted(ref, puntaje, side="right") / len(ref)


# ----------------------------- CONFIGURACIÓN DE LA PÁGINA -----------------------------
st.set_page_config(
    page_title="Evaluación de Riesgo de Crédito (PD, LGD, E[L])",
    page_icon="💳",
    layout="wide",
)

sc, red_pd, prep_lgd, red_lgd = cargar()

st.title("Evaluación Integral de Riesgo de Crédito")
st.write(
    f"Herramienta basada en **Redes Neuronales Artificiales** (Lending Club 2007–2015) para estimar "
    f"la **Probabilidad de Incumplimiento (PD)**, el **Puntaje en Scorecard**, la **Pérdida Dado el Incumplimiento (LGD)** "
    f"y la **Pérdida Esperada (E[L])** antes de otorgar un crédito."
)

# ----------------------------- BARRA LATERAL (SIDEBAR) -----------------------------
with st.sidebar:
    st.header("Documentación y Enlaces")
    if REPORTE_URL:
        st.link_button("📖 Reporte Técnico (Blog)", REPORTE_URL)
    else:
        st.caption("📖 Reporte técnico: pendiente de enlace")

    if VIDEO_URL:
        st.link_button("🎥 Video Promocional", VIDEO_URL)
    else:
        st.caption("🎥 Video promocional: pendiente de enlace")

    st.divider()
    st.subheader("Modelos Desplegados")
    st.markdown(
        """
        * **PD (Clasificación)**: MLP PyTorch de 1,281 parámetros con activación Tanh.
        * **Scorecard**: Base 600 pts, Odds 5:1, PDO 20 pts.
        * **LGD (Regresión)**: MLP PyTorch de 1,281 parámetros con salida Sigmoide $[0, 1]$.
        * **Pérdida Esperada**:
          $$\\mathbb{E}[L] = \\text{PD} \\cdot \\text{LGD} \\cdot \\text{EAD}$$
        """
    )
    st.caption("Datos estandarizados sobre 38 variables ex-ante sin fuga de información post-crédito.")

# ----------------------------- 1. FORMULARIO DE ENTRADA -----------------------------
with st.form("solicitud"):
    st.subheader("1. Solicitud de Crédito y Perfil Financiero")
    c1, c2, c3 = st.columns(3)

    with c1:
        loan_amnt = st.number_input("Monto solicitado (USD, EAD)", 500, 40000, 10000, 500)
        term_60m = int(
            st.radio("Plazo", [36, 60], horizontal=True, format_func=lambda m: f"{m} meses") == 60
        )
        purpose = st.selectbox(
            "Propósito del crédito",
            list(PROPOSITOS),
            index=list(PROPOSITOS).index("debt_consolidation"),
            format_func=PROPOSITOS.get,
        )
        annual_inc = st.number_input("Ingreso anual (USD)", 4000, 500000, 60000, 1000)
        verification_status = st.selectbox(
            "Verificación del ingreso",
            list(VERIFICACION),
            format_func=VERIFICACION.get,
        )

    with c2:
        home_ownership = st.selectbox("Vivienda", list(VIVIENDA), index=1, format_func=VIVIENDA.get)
        dti = st.slider(
            "Deuda / Ingreso (DTI, %)",
            0.0,
            40.0,
            15.0,
            0.5,
            help="Pagos mensuales de deuda existentes sobre el ingreso mensual",
        )
        emp_sin = st.checkbox("No informo antigüedad laboral")
        emp_length_anios = np.nan if emp_sin else st.slider("Antigüedad laboral (años)", 0, 10, 5)
        antiguedad_hist_anios = st.slider(
            "Antigüedad del historial crediticio (años)", 0.0, 50.0, 12.0, 0.5
        )
        total_acc = st.slider("Cuentas de crédito en total", 1, 80, 21)

    with c3:
        open_acc = st.slider("Cuentas de crédito abiertas", 0, 40, 9)
        revol_bal = st.number_input("Saldo rotativo en tarjetas (USD)", 0, 200000, 9500, 500)
        revol_util = st.slider("Uso del crédito rotativo (%)", 0.0, 120.0, 56.0, 1.0)
        inq_last_6mths = st.slider("Consultas recientes al buró (últimos 6 meses)", 0, 8, 1)
        delinq_2yrs = st.slider("Moras de 30+ días (últimos 2 años)", 0, 10, 0)
        ind_mora_previa = int(st.checkbox("He tenido moras previas en mi historial"))
        pub_rec = st.slider("Registros públicos negativos", 0, 5, 0)
        ind_registro_publico = int(pub_rec > 0)

    calcular = st.form_submit_button("Evaluar Solicitud", type="primary")

if calcular:
    st.session_state["perfil"] = dict(
        loan_amnt=loan_amnt,
        term_60m=term_60m,
        purpose=purpose,
        annual_inc=annual_inc,
        verification_status=verification_status,
        home_ownership=home_ownership,
        dti=dti,
        emp_length_anios=emp_length_anios,
        antiguedad_hist_anios=antiguedad_hist_anios,
        total_acc=total_acc,
        open_acc=open_acc,
        revol_bal=revol_bal,
        revol_util=revol_util,
        inq_last_6mths=inq_last_6mths,
        delinq_2yrs=delinq_2yrs,
        ind_mora_previa=ind_mora_previa,
        pub_rec=pub_rec,
        ind_registro_publico=ind_registro_publico,
    )

perfil = st.session_state.get("perfil")
if perfil is None:
    st.info("Completa el formulario y presiona **Evaluar Solicitud** para ver los resultados.")
    st.stop()
else:
    # ----------------------------- 2. RESULTADOS INTEGRALES -----------------------------
    r = evaluar(perfil, sc, red_pd, prep_lgd, red_lgd, con_puntos=True)
    pct = percentil(r["puntaje"], sc)

    st.subheader("2. Resultados de la Evaluación de Riesgo")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Puntaje Scorecard", f"{r['puntaje']:.0f} pts", help="Escala centrada en 600 puntos. Más puntos = Menor riesgo.")
    m2.metric("Probabilidad de Incumplimiento (PD)", f"{100 * r['pd']:.1f} %", help="Probabilidad de que el cliente caiga en mora grave.")
    m3.metric("Pérdida Dado Incumplimiento (LGD)", f"{100 * r['lgd']:.1f} %", help="Porcentaje del crédito no recuperado tras mora y cobranza.")
    m4.metric("Pérdida Esperada E[L]", f"${r['el']:,.2f} USD", help="E[L] = PD * LGD * EAD (Monto solicitado)")

    st.write(
        f"El **{pct:.0f} %** de los créditos históricos de Lending Club tiene un puntaje igual o menor al tuyo "
        f"(mayor o igual riesgo); el **{100 - pct:.0f} %** restante tiene un puntaje superior."
    )

    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.markdown("**Posición en la Distribución Histórica de Puntajes**")
        hist, bordes = np.histogram(sc["puntajes_referencia"], bins=40)
        datos_hist = pd.DataFrame({"puntaje": bordes[:-1], "clientes": hist})
        barras = (
            alt.Chart(datos_hist)
            .mark_bar(opacity=0.6)
            .encode(
                x=alt.X("puntaje:Q", bin=False, title="Puntaje"),
                y=alt.Y("clientes:Q", title="Clientes históricos"),
            )
        )
        linea = (
            alt.Chart(pd.DataFrame({"x": [r["puntaje"]]}))
            .mark_rule(color="red", size=3)
            .encode(x="x:Q")
        )
        st.altair_chart(barras + linea, use_container_width=True)

    with col_der:
        st.markdown("**Atribución de Puntos por Variable (Gradientes Integrados)**")
        p = r["puntos"]
        top = p.reindex(p.abs().sort_values(ascending=False).index[:8]).sort_values()
        st.caption(
            f"Impacto relativo frente al cliente promedio ({sc['puntos_base']:.0f} pts). "
            "Valores positivos suman puntos (bajan el riesgo); negativos restan puntos."
        )
        st.bar_chart(pd.DataFrame({"puntos": top}), horizontal=True)

    # ----------------------------- 3. SIMULADOR EN TIEMPO REAL -----------------------------
    st.subheader("3. Simulador 'What-If': ¿Cómo mejorar las condiciones?")
    st.caption("Ajusta los parámetros financieros para ver el impacto inmediato en el Puntaje, la PD, la LGD y la Pérdida Esperada.")

    s1, s2, s3, s4 = st.columns(4)
    f_dti = s1.slider("Variar Deuda/Ingreso (DTI)", -50, 50, 0, 5, format="%d %%")
    f_inc = s2.slider("Variar Ingreso Anual", -50, 50, 0, 5, format="%d %%")
    f_monto = s3.slider("Variar Monto Solicitado", -50, 50, 0, 5, format="%d %%")
    f_util = s4.slider("Variar Uso de Tarjetas", -50, 50, 0, 5, format="%d %%")

    plazo_sim = st.radio(
        "Simular Plazo",
        ["igual", 36, 60],
        horizontal=True,
        format_func=lambda v: "El mismo seleccionado" if v == "igual" else f"{v} meses",
    )

    nuevo_perfil = dict(
        perfil,
        dti=perfil["dti"] * (1 + f_dti / 100),
        annual_inc=perfil["annual_inc"] * (1 + f_inc / 100),
        loan_amnt=perfil["loan_amnt"] * (1 + f_monto / 100),
        revol_util=perfil["revol_util"] * (1 + f_util / 100),
    )
    if plazo_sim != "igual":
        nuevo_perfil["term_60m"] = int(plazo_sim == 60)

    rn = evaluar(nuevo_perfil, sc, red_pd, prep_lgd, red_lgd)

    n1, n2, n3, n4 = st.columns(4)
    delta_pts = rn["puntaje"] - r["puntaje"]
    delta_pd = (rn["pd"] - r["pd"]) * 100
    delta_lgd = (rn["lgd"] - r["lgd"]) * 100
    delta_el = rn["el"] - r["el"]

    n1.metric("Puntaje Simulado", f"{rn['puntaje']:.0f} pts", f"{delta_pts:+.1f} pts")
    n2.metric("PD Simulada", f"{100 * rn['pd']:.1f} %", f"{delta_pd:+.1f} pp", delta_color="inverse")
    n3.metric("LGD Simulada", f"{100 * rn['lgd']:.1f} %", f"{delta_lgd:+.1f} pp", delta_color="inverse")
    n4.metric("E[L] Simulada", f"${rn['el']:,.2f} USD", f"${delta_el:+,.2f} USD", delta_color="inverse")

    st.caption("Simulación basada en los modelos neuronales de Credit Risk Analysis (Lending Club); herramienta con fines analíticos y educativos.")
