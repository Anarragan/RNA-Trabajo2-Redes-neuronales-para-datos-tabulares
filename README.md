# RNA-Trabajo2-Redes-neuronales-para-datos-tabulares


## Cómo ejecutar

```bash
python -m venv .venv && .venv\Scripts\activate          # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

cd Parte_1
python main.py todo        # flujo completo (~5 min): rl, rna, evaluar, scorecard, comparar
python main.py todo --demo # versión rápida (< 2 min): rl, rna y evaluar
python main.py --help      # lista de comandos con ejemplos
python main.py rna --help  # opciones de un comando
```

## Aplicación web (scorecard)

```bash
streamlit run app.py       # desde la raíz del repositorio; abre http://localhost:8501
```

Carga `Parte_1/modelos/scorecard.joblib` y `red_neuronal.pt` (ya en el repositorio): no reentrena ni usa internet. 
Entrega puntaje, PD, percentil frente a la población histórica y un simulador de cambios. 
Para cambiar la scorecard: `python main.py scorecard --base 600 --odds 5 --pdo 20` y reiniciar la app. 
Los enlaces al reporte y al video se pegan en `REPORTE_URL` y `VIDEO_URL` al inicio de `app.py`.

**Despliegue gratis (Streamlit Community Cloud):** subir el repositorio a GitHub, entrar a share.streamlit.io, 
*Create app* → elegir repositorio y rama, *Main file path* = `app.py`, Python 3.12 en *Advanced settings* → Deploy. 
Instala solo `requirements.txt` y no necesita los datos grandes (`data/`).

## Orden de ejecución y qué genera cada comando

Todo se lanza desde `Parte_1/` con `python main.py <comando> [opciones]`. Las opciones se pasan al script del comando.

| # | Comando | Script | Lee | Escribe |
|---|---|---|---|---|
| 0 | `datos` (solo si se quiere rehacer; necesita `loan.csv` de Kaggle) | `preparacion_datos.py` | `loan.csv` | `data/datos_modelo.csv`, `data/base_lgd.csv` |
| 1 | `rl` regresión logística (T3; busca `C` en 10 valores) | `regresion_logistica.py` | `data/datos_modelo.csv` | `modelos/regresion_logistica.joblib`, `resultados/t3_busqueda.csv`, `t3_resumen.csv` |
| 2 | `rna` red neuronal (T4) | `red_neuronal.py` | `data/datos_modelo.csv` | `modelos/red_neuronal.pt`, `resultados/t4_busqueda.csv`, `t4_semillas.csv` (el resumen se imprime en consola) |
| 3 | `evaluar` (T5) | `evaluacion.py` | datos + los dos modelos | `resultados/t5_metricas.csv`, `t5_particiones.csv`, `figuras/t5_calibracion.png`, `t5_roc.png` |
| 4 | `scorecard` (T6) | `scorecard_red_neuronal.py` | datos + `red_neuronal.pt` | `modelos/scorecard.joblib`, `resultados/t6_rn_puntos_variables.csv` |
| 5 | `comparar` | `comparar.py` | `t3_resumen.csv`, `t4_*.csv`, `red_neuronal.pt` | `resultados/comparacion.csv` (una sola tabla: configuraciones, tiempo, parámetros, semillas, media/desv/mejor/peor) |

Comparación justa: ambos modelos usan las mismas particiones y el mismo preprocesamiento, prueban 10 configuraciones elegidas por pérdida de validación y reportan tiempo de búsqueda y de entrenamiento final por separado (`comparar`).

`todo` corre 1 a 5 (no incluye `datos`). Con `--demo`, `rna` y `evaluar` escriben archivos `*_demo.*` (ignorados por git) y no pisan los reales.
Los datos procesados y los modelos entrenados están en el repositorio: no hace falta internet ni reentrenar para evaluar.
Semillas: la partición es temporal (sin azar), la búsqueda usa la semilla 0 y la red final las semillas 0..29.

Apoyo (no se ejecutan solos): `preprocesamiento.py` (partición temporal + estandarización, igual para todos los modelos; también lo usa `Parte_2`) y `metricas.py`.

## Carpetas

```
Parte_1/  scripts de arriba · data/ (datos procesados) · modelos/ (modelos entrenados) · resultados/ (tablas) · figuras/
Parte_2/  pérdida dado incumplimiento (importa de Parte_1: no mover archivos de Parte_1)
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
