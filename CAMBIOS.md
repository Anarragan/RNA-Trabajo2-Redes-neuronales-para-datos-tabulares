# CAMBIO

## Por qué cambian los resultados de la red (importante)

La red anterior **no era comparable** con la regresión logística:

1. **Fuga de información.** Solo borraba algunas columnas y conservaba `total_pymnt`, `recoveries`, `out_prncp`,
   `last_pymnt_amnt`, `total_rec_prncp`, `int_rate`, `grade` y `sub_grade`: se conocen después de dar el crédito o son
   decisiones de la plataforma. Por eso reportaba 99,79 % de precisión (84 errores en 40 280 créditos) y pérdidas de
   validación de ~0,01. Con `Parte_1` (solo información disponible al solicitar) el AUC de prueba es ≈ 0,66.
2. **Partición aleatoria** en vez de temporal (la guía exige temporal).
3. **172 columnas de entrada** (por `get_dummies` sobre `sub_grade`, `addr_state`...) frente a 34 en `Parte_1`.

Los números de la red antigua **no deben usarse** en el reporte. Los de la regresión logística (Tareas 1-3) **no cambiaron**:
se comparó la salida completa (particiones, métricas y coeficientes) antes y después, y es idéntica; además
`datos_modelo.csv` se regenera idéntico byte a byte desde el `loan.csv` original.

## Qué se optimizó (y cuánto)

Tiempo medido en la misma máquina (CPU, 8 hilos):

| | Red antigua (`datos.py`) | Red nueva (`red_neuronal.py`) |
|---|---|---|
| Qué hace | 10 configuraciones, 1 entrenamiento final | 10 configuraciones + **30 semillas** (lo que exige la guía) |
| Tiempo total | **1504 s (25 min)** | **288 s (4,8 min)** |
| Modo demo | no existe | 21 s |

Por qué tardaba la red antigua (medido, no supuesto):
- **Un solo paso de gradiente por época.** Procesaba las 188 mil filas de golpe, así que 250 épocas eran solo 250 pasos.
  Con la red [128, 64, 32] sobre datos limpios, 20 épocas de ese modo dan AUC de validación 0,51 (no aprende); con
  mini-lotes de 1024 (~88 pasos por época) llega a 0,672 en 20 épocas. Mini-lotes → converge en 10-25 épocas, no en 250.
- **Cada paso era caro:** 188 mil filas × 172 columnas (≈ 1,2 s por época en la red de 3 capas). Ahora son 90 mil filas × 34.
- **Redes de 3 capas con 250 épocas** consumían 300 s por intento; 7 de los 10 intentos agotaron las 250 épocas.
- **Descarga, lectura del Excel y limpieza en cada ejecución** (23,5 s) y **ningún modelo guardado**.
- **El número de hilos no influye** (1 a 16 hilos: ~750 ms por época): las matrices son pequeñas y domina la sobrecarga de Python.

Cambios que implican decisiones (avísenme si alguno no les gusta):
- `nn.Sigmoid` + `BCELoss` → `BCEWithLogitsLoss`: es la misma función de pérdida, pero estable numéricamente; la probabilidad es `sigmoide(logit)`.
- Máximo 40 épocas con paciencia 5 (antes 250 y paciencia 10).
- Se agregó una **semilla** (antes ninguna), no se repiten configuraciones en la búsqueda, y se reportan parámetros y tiempos.
- Se mantienen: MLP con ReLU, dropout, L2 (`weight_decay` de Adam), Adam, parada temprana y búsqueda aleatoria de 10 configuraciones.

## Pendiente

Tareas 1-2 (caso de uso y análisis descriptivo, son texto y figuras del reporte), Tareas 6-7 (scorecard e
interpretabilidad) y toda la Parte 2. La Tarea 5 está hecha con AUC, Gini, KS, Brier y curva de calibración;
falta incluir las figuras numeradas en el reporte.
