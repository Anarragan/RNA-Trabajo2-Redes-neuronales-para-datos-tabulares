# RNA-Trabajo2-Redes-neuronales-para-datos-tabulares


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
