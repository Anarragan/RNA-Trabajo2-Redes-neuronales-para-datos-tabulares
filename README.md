# RNA-Trabajo2-Redes-neuronales-para-datos-tabulares


## Cómo ejecutar

```bash
python -m venv .venv && .venv\Scripts\activate          # (Linux/Mac: source .venv/bin/activate)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cd Parte_1
python main.py --demo      # todo el flujo en versión rápida (< 2 min)
python main.py             # flujo completo (~5 min)
python main.py --solo t4   # un paso: datos | t2 | t3 | t4 | t5
python red_neuronal.py --busqueda 0 --capas 64 32 --lr 0.001 --dropout 0.2 --l2 1e-4 --semillas 5   # parámetros a mano
```

Los datos procesados ya están en el repositorio: no hace falta internet. Solo `--con-datos` los rehace desde `loan.csv`.
Semillas: la partición es temporal (sin azar), la búsqueda usa la semilla 0 y la red final las semillas 0..29.

## Qué hace cada archivo

| Paso | Script | Lee | Escribe |
|---|---|---|---|
| datos | `preparacion_datos.py` | `loan.csv` (Kaggle) | `data/datos_modelo.csv`, `data/base_lgd.csv` |
| T2 hipótesis | `analisis_descriptivo.py` | `datos_modelo.csv` | `resultados/t2_hipotesis.csv`, `figuras/t2_hipotesis.png` |
| T3 regresión logística | `regresion_logistica.py` | `datos_modelo.csv` | `modelos/regresion_logistica.joblib` |
| T4 red neuronal | `red_neuronal.py` | `datos_modelo.csv` | `modelos/red_neuronal.pt`, `resultados/t4_*` |
| T5 evaluación | `evaluacion.py` | datos + los dos modelos | `resultados/t5_*`, `figuras/t5_*` |

Apoyo: `preprocesamiento.py` (partición temporal + estandarización, igual para todos los modelos) y `metricas.py`.

**Carpeta `modelos/`**: modelos ya entrenados. No se usan a mano: `evaluacion.py` los carga (y la scorecard y la app web los cargarán)
con `joblib.load("modelos/regresion_logistica.joblib")` y `red_neuronal.cargar_red()`. Así nadie reentrena.

**Carpeta `resultados/`** (alimentan las tablas del reporte):
`t2_hipotesis.csv` veredicto de H1-H6 · `t4_busqueda.csv` las 10 configuraciones probadas ·
`t4_semillas.csv` métricas de las 30 semillas · `t4_resumen.txt` media/desv/mejor/peor, parámetros y tiempo ·
`t5_metricas.csv` AUC/Gini/KS/Brier de ambos modelos · `t5_particiones.csv` tamaño y periodo de cada partición.

`Parte_2/` está vacía (ver su README). `legacy/` es el código anterior, solo de consulta. `CAMBIOS.md` explica qué se movió.
Caso de uso e hipótesis: `CASO_DE_USO.md`.

Pendiente de la Parte 1: scorecard y variables de riesgo (T6-T7).

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
