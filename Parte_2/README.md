# Parte 2: Pérdida dado incumplimiento (LGD) y pérdida esperada

Esta carpeta está vacía a propósito: aquí van los scripts de la Parte 2. La Parte 1 ya deja lista la base.
Sigan el mismo patrón de la Parte 1: un script por tarea, con argumentos de línea de comandos, entrada y salida en archivos.

## Qué ya está listo (no hay que volver a limpiar datos)

| Necesitan | Dónde está | Cómo usarlo |
|---|---|---|
| Créditos malos con sus campos de recuperación | `../Parte_1/data/base_lgd.csv` (58 819 filas) | `pd.read_csv(..., parse_dates=["issue_d"])` |
| Los mismos datos y partición temporal de la Parte 1 | `../Parte_1/preprocesamiento.py` | `from preprocesamiento import partir, Preprocesador` |
| Probabilidad de incumplimiento (PD) ya entrenada | `../Parte_1/modelos/regresion_logistica.joblib` y `red_neuronal.pt` | `joblib.load(...)` / `red_neuronal.cargar_red()` + `predecir()` |
| Métricas y tabla de resultados | `../Parte_1/metricas.py` | `evaluar(y, p)` |
| Red neuronal con búsqueda y N semillas | `../Parte_1/red_neuronal.py` | Se puede copiar como plantilla: cambiar la salida a sigmoide + pérdida para una fracción en [0, 1] |

`base_lgd.csv` tiene las 20 columnas del modelo de la Parte 1 (más `malo`) y estas columnas de recuperación:
`id, loan_status, funded_amnt, out_prncp, total_pymnt, total_rec_prncp, total_rec_int, total_rec_late_fee,
recoveries, collection_recovery_fee, last_pymnt_d`.
**Estas columnas son posteriores al crédito**: sirven para construir la variable objetivo (LGD), **nunca como predictoras**
(sería la misma fuga de información que tenía la red anterior, ver `../CAMBIOS.md`).

## Lo que falta (de la guía)

1. LGD: fórmula, justificación de cada campo y limitaciones (cuidado: `recoveries` y `total_rec_prncp` tienen
   definiciones distintas en el diccionario de Lending Club; léanlas en el diccionario, no en una IA).
2. Modelo de referencia (por ejemplo, regresión lineal).
3. Red neuronal para la LGD: justificar pérdida y activación de salida según la naturaleza de la variable objetivo.
4. Evaluación: MAE, RMSE y comparación de las distribuciones predicha y observada.
5. Pérdida esperada `E[L] = PD · LGD · EAD`, definiendo la EAD.
6. Las mismas reglas de comparación justa de la Parte 1: mismas particiones, presupuesto de búsqueda,
   30 semillas y número de parámetros.
