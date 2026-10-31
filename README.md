# RNA-Trabajo2-Redes-neuronales-para-datos-tabulares


## Cómo ejecutar

```bash
python -m venv .venv && .venv\Scripts\activate          # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# --- PARTE 1: Probabilidad de Incumplimiento (PD) ---
cd Parte_1
python main.py todo        # flujo completo (~5 min): rl, rna, evaluar, scorecard, comparar
python main.py todo --demo # versión rápida (< 2 min): rl, rna y evaluar
python main.py --help      # lista de comandos con ejemplos
python main.py rna --help  # opciones de la red de PD (capas, lr, dropout, etc.)
cd ..

# --- PARTE 2: Pérdida Dado el Incumplimiento (LGD) y Pérdida Esperada E[L] ---
cd Parte_2
python main.py todo        # flujo completo (~3-4 min): datos, rl, rna, evaluar
python main.py todo --demo # versión rápida (< 1 min): rna sobre 3 semillas
python main.py --help      # lista de comandos con ejemplos
python main.py rna --help  # opciones de la red de LGD (capas, lr, dropout, etc.)
cd ..
```

## Aplicación web (scorecard y LGD integrada)

```bash
streamlit run app.py       # desde la raíz del repositorio; abre http://localhost:8501
```

Carga `Parte_1/modelos/scorecard.joblib`, `red_neuronal.pt` y `Parte_2/modelos/red_neuronal_lgd.pt`: no reentrena ni usa internet. 
Entrega puntaje, PD, LGD, Pérdida Esperada $\mathbb{E}[L]$, percentil frente a la población histórica y un simulador "What-If" de cambios. 
Para cambiar la scorecard: `python main.py scorecard --base 600 --odds 5 --pdo 20` y reiniciar la app. 
Los enlaces al reporte y al video se pegan en `REPORTE_URL` y `VIDEO_URL` al inicio de `app.py`.

**Despliegue gratis (Streamlit Community Cloud):** subir el repositorio a GitHub, entrar a share.streamlit.io, 
*Create app* → elegir repositorio y rama, *Main file path* = `app.py`, Python 3.12 en *Advanced settings* → Deploy. 
Instala solo `requirements.txt` y no necesita los datos grandes (`data/`).

## Modificación de parámetros sin editar código (Línea de comandos y Archivo de Configuración)

De acuerdo con el requisito de sustentación en vivo (**Punto 3 de la guía**), todos los hiperparámetros pueden modificarse sin tocar el código fuente, ya sea mediante argumentos CLI o mediante archivos de configuración JSON.

### 1. Parámetros de la Red Neuronal de Clasificación (Parte 1: PD)

Desde la carpeta `Parte_1/`:
```bash
# A. Búsqueda por etapas automática (oficial):
python main.py rna

# B. Configuración manual directa por CLI:
python main.py rna --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4 --activacion tanh --epocas 50 --semillas 5

# C. Desde archivo de configuración JSON:
python main.py rna --config config.json

# D. Demostración rápida (< 30 s para sustentación):
python main.py rna --demo
```

### 2. Parámetros de la Red Neuronal de Regresión (Parte 2: LGD)

Desde la carpeta `Parte_2/`:
```bash
# A. Búsqueda por etapas automática (oficial):
python main.py rna

# B. Configuración manual directa por CLI:
python main.py rna --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4 --activacion leaky_relu --epocas 40 --semillas 5

# C. Desde archivo de configuración JSON:
python main.py rna --config config.json

# D. Demostración rápida (< 30 s para sustentación):
python main.py rna --demo
```

### 3. Parámetros del Scorecard y Modelos de Referencia

```bash
# Calibrar escala del Scorecard (Parte 1):
python main.py scorecard --base 600 --odds 5 --pdo 20
python main.py scorecard --config config.json

# Penalización L2 de Regresión Logística (Parte 1, C = 1 / lambda):
python main.py rl --C 0.1
python main.py rl --config config.json

# Penalización L2 de Regresión Ridge (Parte 2, alpha):
python main.py rl --alpha 1.0   # sin --alpha se busca con el mismo presupuesto que la red
python main.py rl --config config.json
```

### Tabla de Hiperparámetros Disponibles

| Parámetro | Argumento CLI | Clave JSON | Valores permitidos / Ejemplos | Defecto Parte 1 | Defecto Parte 2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arquitectura** | `--capas` | `"capas"` | Lista de enteros (ej. `64 32`) | `[32, 16]` | `[32, 16]` |
| **Tasa de aprendizaje** | `--lr` | `"lr"` | Decimal (ej. `0.001`, `5e-4`) | `1e-3` | `1e-3` |
| **Dropout (regularización)**| `--dropout` | `"dropout"` | Decimal en $[0, 1)$ (ej. `0.2`) | `0.2` | `0.2` |
| **Weight Decay / L2** | `--l2` | `"l2"` | Decimal (ej. `1e-4`, `1e-5`) | `1e-4` | `1e-4` |
| **Función de activación** | `--activacion` | `"activacion"` | `relu`, `tanh` (LGD: `leaky_relu`) | `relu` | `relu` |
| **Optimizador** | `--optimizador`| `"optimizador"`| `adam`, `sgd` | `adam` | `adam` |
| **Máximo de épocas** | `--epocas` | `"epocas"` | Entero positivo (ej. `50`) | `60` (`10` en demo) | `40` (`10` en demo) |
| **Semillas independientes**| `--semillas` | `"semillas"` | Entero positivo (ej. `30`) | `30` (`3` en demo) | `30` (`3` en demo) |
| **Parada temprana** | `--paciencia` | `"paciencia"` | Entero positivo | `10` | `10` |
| **Puntaje Base** | `--base` | `"base"` | Número real (ej. `600`) | `600` | N/A |
| **Odds de referencia** | `--odds` | `"odds"` | Número real (ej. `5` para 5:1) | `5` | N/A |
| **Puntos duplicar odds**| `--pdo` | `"pdo"` | Número real (ej. `20`) | `20` | N/A |

### Estructura de Ejemplo de `config.json`
```json
{
  "capas": [64, 32],
  "activacion": "relu",
  "dropout": 0.2,
  "l2": 0.0001,
  "optimizador": "adam",
  "lr": 0.001,
  "epocas": 40,
  "semillas": 5,
  "paciencia": 10
}
```

---

## Orden de ejecución y qué genera cada comando

### Parte 1: Probabilidad de Incumplimiento (`Parte_1/`)

Todo se lanza desde `Parte_1/` con `python main.py <comando> [opciones]`. Las opciones se pasan al script del comando.

| # | Comando | Script | Lee | Escribe |
|---|---|---|---|---|
| 0 | `datos` (solo si se quiere rehacer; necesita `loan.csv` de Kaggle) | `preparacion_datos.py` | `loan.csv` | `data/datos_modelo.csv`, `data/base_lgd.csv` |
| 1 | `rl` regresión logística (T3; busca `C` con el mismo número de configuraciones que la red) | `regresion_logistica.py` | `data/datos_modelo.csv` | `modelos/regresion_logistica.joblib`, `resultados/t3_busqueda.csv`, `t3_resumen.csv` |
| 2 | `rna` red neuronal (T4) | `red_neuronal.py` | `data/datos_modelo.csv` | `modelos/red_neuronal.pt`, `resultados/t4_busqueda.csv`, `t4_semillas.csv` (el resumen se imprime en consola) |
| 3 | `evaluar` (T5) | `evaluacion.py` | datos + los dos modelos | `resultados/t5_metricas.csv`, `t5_particiones.csv`, `figuras/t5_calibracion.png`, `t5_roc.png` |
| 4 | `scorecard` (T6) | `scorecard_red_neuronal.py` | datos + `red_neuronal.pt` | `modelos/scorecard.joblib`, `resultados/t6_rn_puntos_variables.csv` |
| 5 | `comparar` | `comparar.py` | `t3_resumen.csv`, `t4_*.csv`, `red_neuronal.pt` | `resultados/comparacion.csv` (una sola tabla: configuraciones, tiempo, parámetros, semillas, media/desv/mejor/peor) |

Comparación justa: ambos modelos usan las mismas particiones y el mismo preprocesamiento, prueban el mismo número de configuraciones (el de la búsqueda de la red) elegidas por pérdida de validación y reportan tiempo de búsqueda y de entrenamiento final por separado (`comparar`).

`todo` corre 1 a 5 (no incluye `datos`). Con `--demo`, `rna` y `evaluar` escriben archivos `*_demo.*` (ignorados por git) y no pisan los reales.

### Parte 2: Pérdida Dado el Incumplimiento y Pérdida Esperada (`Parte_2/`)

Todo se lanza desde `Parte_2/` con `python main.py <comando> [opciones]`.

| # | Comando | Script | Lee | Escribe |
|---|---|---|---|---|
| 0 | `datos` construcción objetivo LGD | `construccion_lgd.py` | `../Parte_1/data/base_lgd.csv` | `resultados/lgd_procesado.csv` |
| 1 | `rl` regresión lineal Ridge acotada (T2; busca `alpha` con el mismo número de configuraciones que la red, por MSE de validación) | `regresion_lineal.py` | `resultados/lgd_procesado.csv` | `modelos/regresion_lineal.joblib`, `modelos/preprocesador_lgd.joblib`, `resultados/t2_busqueda_ridge.csv`, `t2_resumen_ridge.csv` |
| 2 | `rna` red neuronal MLP para LGD (T3) | `red_neuronal_lgd.py` | `resultados/lgd_procesado.csv` | `modelos/red_neuronal_lgd.pt`, `resultados/t4_resumen.txt` |
| 3 | `evaluar` y Pérdida Esperada (T4-T5) | `evaluacion.py` | datos + modelos | `figuras/t4_distribucion_lgd.png`, `resultados/t4_predicciones_lgd.csv`, `resultados/t5_perdida_esperada.csv` |
| 4 | `comparar` | `comparar.py` | datos + modelos + búsquedas | `resultados/comparacion_lgd.csv` (configuraciones, tiempos, parámetros, semillas, media/desv/mejor/peor) |

`todo` corre del 0 al 4 en orden. Con `--demo`, `rna` y `evaluar` corren en versión rápida (< 30 s).

Los datos procesados y los modelos entrenados están en el repositorio: no hace falta internet ni reentrenar para evaluar.
Semillas: la partición es temporal (sin azar), la búsqueda usa la semilla 0 y la red final las semillas 0..29.

Apoyo (no se ejecutan solos): `preprocesamiento.py` (partición temporal + estandarización, igual para todos los modelos; también lo usa `Parte_2`) y `metricas.py`.

## Carpetas

```
Parte_1/  scripts de PD · data/ (datos procesados) · modelos/ · resultados/ · figuras/
Parte_2/  scripts de LGD y E[L] · modelos/ · resultados/ · figuras/
app.py    aplicación web Streamlit interactiva
legacy/   código anterior de la red, solo de consulta
```

---

# Guía inicial de GitFlow Simplificado

> **Spoiler:** El único comando que uso es:
> ```bash
> git merge develop
> ```

---

## Paso 1: Crear tu rama de trabajo

Al clonar el repositorio, por defecto estarás en la rama `main`. Debes crear tu rama propia **a partir de `develop`** (puedes usar las ayudas visuales de la extensión de Git en VS Code).

* **Resultado:** Tendrás una rama propia (`tu-nombre`) basada en `develop`.

---

## Paso 2: Guardar tus cambios locales

Mientras trabajas en tu rama, guarda el avance haciendo un `commit` estándar hacia tu rama personal.

---

## Paso 3: Subir tu tarea y notificar al equipo

Cuando completes tu tarea o ítem:
1. Haz un **Pull Request (PR)** dirigido a la rama `develop`.
2. Notifica a tus compañeros para que sincronicen su rama local con el nuevo `develop`.

> ⚠️ **Aclaración importante para los PR:**
> En GitHub, tras hacer *commit* y *push* en tu rama personal, aparecerá un recuadro naranja con el botón **"Compare & pull request"**.
> 
> Al hacer clic, **asegúrate de verificar las ramas de destino:**
> * **`base`:** `develop`
> * **`compare`:** `tu-nombre`

---

## ¿Cómo actualizar tu rama con los cambios de `develop`?

Cuando un compañero notifique que subió cambios a `develop`:

1. Ve a la rama `develop` local en VS Code.
2. Haz un **Fetch / Pull** para traer los cambios remotos.
3. Cambia nuevamente a tu rama personal (`tu-nombre`).
4. Ejecuta en la terminal:
   ```bash
   git merge develop
   ```
* **Resultado:** Tu rama quedará completamente sincronizada con `develop`.

---

## ¿Qué pasa cuando actualizan `develop` y tengo cambios sin guardar?

Si alguien actualizó `develop`, tienes cambios en una versión antigua y **no los quieres perder**, puedes usar el **Stash** para guardar una especie de *checkpoint*.

### Pasos en VS Code:
1. En el menú lateral izquierdo de Git (Control de código fuente), haz clic en los **tres puntos (`...`)**.
2. Ve a **Stash** $\rightarrow$ **Stash (Include Untracked)** y asígnale un nombre a tu trabajo.
3. Actualiza tu rama `develop` local (como se indicó en la sección anterior).
4. Cambia a tu rama personal y ejecuta `git merge develop`.
5. Ve nuevamente a los tres puntos (`...`) de la interfaz gráfica de Git:
   * Busca **Stash** $\rightarrow$ **Apply Latest Stash**.
6. Si hay **conflictos**, resuélvelos decidiendo qué cambios entrantes (*Incoming*) mantienes o cuáles de tus cambios (*Current*) prefieres conservar.

### Al terminar el trabajo
Se actualiza main con la version final de develop, se sigue este flujo para poder mantener un correcto manejo de versiones.
