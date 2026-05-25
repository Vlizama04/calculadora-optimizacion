import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import line_search

st.set_page_config(page_title="Calculadora de Optimización", layout="wide")

st.title("Calculadora de Optimización No Lineal")

# --- INSTRUCCIONES DE USO ---
with st.expander("📖 Guía de uso y sintaxis matemática (Haz clic para expandir/contraer)", expanded=True):
    st.markdown("""
    **Sintaxis obligatoria para la Función Objetivo:**
    * **Variables:** Utiliza siempre el formato `x1`, `x2`, `x3`, etc. (No uses `x` o `y` de forma aislada).
    * **Potencias:** Puedes usar `^` o `**` (Ejemplo: `x1^2` o `x1**2`).
    * **Multiplicación:** Usa explícitamente el asterisco `*` (Ejemplo: `2*x1`, **no** `2x1`).
    * **Exponencial (Euler):** Para $e^{x_1}$, escribe `exp(x1)` o `e^(x1)`.
    * **Trigonometría y otras:** Puedes usar `sin(x1)`, `cos(x1)`, `log(x1)`.

    *Ejemplo de función compleja válida de 2 variables: `(x1 - 1)^2 + exp(x2^2)`*

    ---
    **Métodos disponibles:**
    * **Gradiente:** Desciende en la dirección opuesta al gradiente. Convergencia garantizada pero puede ser lenta cerca del mínimo.
    * **Gradiente Conjugado (Fletcher-Reeves):** Mejora el gradiente usando información de la dirección anterior. Se reinicia cada `n` iteraciones (donde `n` es el número de variables) para mantener estabilidad numérica.
    * **Newton:** Usa el Hessiano para una convergencia cuadrática. Si la dirección obtenida no es de descenso (Hessiano indefinido), se cae automáticamente al gradiente negativo para garantizar progreso.

    ---
    **Condiciones de Wolfe:**
    * **c1** controla la condición de suficiente descenso (Armijo). Valor típico: `1e-4`.
    * **c2** controla la condición de curvatura. Valor típico: `0.9` (métodos de gradiente) o `0.1` (Newton).
    * Se requiere que `0 < c1 < c2 < 1`.
    * Si la búsqueda de línea no encuentra un paso válido, se usa un paso fijo pequeño de respaldo (`α = 1e-4`) y se notifica al usuario.

    ---
    **Gráficos disponibles:**
    * **Convergencia:** Muestra la norma del gradiente por iteración en escala logarítmica.
    * **Trayectoria 2D** *(solo para funciones de 2 variables):* Muestra el camino recorrido por el método sobre las curvas de nivel de la función.
    * **Tabla de iteraciones:** Detalla el valor de `f(x)`, la norma del gradiente y el paso `α` en cada iteración.
    """)

# --- 1. DATOS DE ENTRADA ---
st.sidebar.header("Datos de Entrada")

num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=50, value=2, step=1)
metodo = st.sidebar.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
func_str = st.sidebar.text_input("Función objetivo", "x1**2 + x2**2")

punto_partida_str = st.sidebar.text_input("Punto de partida (separado por comas)", "2.0, 2.0")
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)

tol = st.sidebar.number_input("Tolerancia de convergencia", min_value=1e-10, value=1e-5, format="%.5f", step=1e-5)

st.sidebar.subheader("Parámetros de Wolfe")
c1 = st.sidebar.number_input("c1 (Suficiente descenso)", min_value=0.0001, max_value=0.9999, value=1e-4, format="%.4f", step=0.0001)
c2 = st.sidebar.number_input("c2 (Curvatura)", min_value=c1, max_value=0.9999, value=0.9, format="%.4f", step=0.1000)

mostrar_tabla = st.sidebar.checkbox("Mostrar tabla de iteraciones", value=False)

# --- 2. PROCESAMIENTO MATEMÁTICO ---
try:
    func_str_procesada = func_str.replace('^', '**')
    vars_simbolicas = sp.symbols(f'x1:{num_vars + 1}')
    f_simbolica = sp.sympify(func_str_procesada, locals={'e': sp.E})

    gradiente_simbolico = [sp.diff(f_simbolica, var) for var in vars_simbolicas]
    hessiano_simbolico = [[sp.diff(g, var) for var in vars_simbolicas] for g in gradiente_simbolico]

    f_num   = sp.lambdify(vars_simbolicas, f_simbolica, 'numpy')
    grad_num = sp.lambdify(vars_simbolicas, gradiente_simbolico, 'numpy')
    hess_num = sp.lambdify(vars_simbolicas, hessiano_simbolico, 'numpy')

    def f_eval(x):    return float(f_num(*x))
    def grad_eval(x): return np.array(grad_num(*x), dtype=float).flatten()
    def hess_eval(x): return np.array(hess_num(*x), dtype=float)

    x0 = np.array([float(v.strip()) for v in punto_partida_str.split(',')])
    if len(x0) != num_vars:
        st.error(f"⚠️ El punto de partida debe tener exactamente {num_vars} componentes separados por comas.")
        st.stop()

except Exception as e:
    st.error(f"⚠️ Error en la sintaxis matemática. Por favor, revisa la guía de uso. Detalle técnico: {e}")
    st.stop()

# --- 3. ALGORITMOS DE OPTIMIZACIÓN ---
if st.button("Ejecutar Optimización"):
    x = x0.copy()
    historial_error = []
    historial_x     = [x.copy()]
    historial_f     = []
    historial_alpha  = []
    avisos_wolfe     = []

    k = 0
    criterio_parada = "Número máximo de iteraciones alcanzado"
    error_actual = np.linalg.norm(grad_eval(x))
    historial_error.append(error_actual)
    historial_f.append(f_eval(x))

    p_previo    = None
    grad_previo = None

    barra_progreso = st.progress(0)

    while k < max_iter and error_actual > tol:
        gk = grad_eval(x)

        # --- Dirección de búsqueda ---
        if metodo == "Gradiente":
            pk = -gk

        elif metodo == "Gradiente Conjugado":
            # Fletcher-Reeves con reinicio cada num_vars iteraciones
            if k == 0 or k % num_vars == 0:
                pk = -gk
            else:
                denom = np.dot(grad_previo, grad_previo)
                if denom < 1e-12:
                    beta = 0.0
                else:
                    beta = np.dot(gk, gk) / denom
                pk = -gk + beta * p_previo
            p_previo    = pk.copy()
            grad_previo = gk.copy()

        elif metodo == "Newton":
            Hk = hess_eval(x)
            try:
                pk = np.linalg.solve(Hk, -gk)
                # Verificar que sea dirección de descenso
                if np.dot(pk, gk) >= 0:
                    pk = -gk
            except np.linalg.LinAlgError:
                pk = -gk

        # --- Búsqueda de línea con condiciones de Wolfe ---
        alpha, _, _, _, _, _ = line_search(
            f_eval, grad_eval, x, pk, gfk=gk, old_fval=f_eval(x), c1=c1, c2=c2
        )

        if alpha is None:
            alpha = 1e-4
            avisos_wolfe.append(k + 1)

        historial_alpha.append(alpha)

        x = x + alpha * pk
        error_actual = np.linalg.norm(grad_eval(x))
        historial_error.append(error_actual)
        historial_f.append(f_eval(x))
        historial_x.append(x.copy())
        k += 1

        barra_progreso.progress(min(k / max_iter, 1.0))

    if error_actual <= tol:
        criterio_parada = "Tolerancia de convergencia alcanzada"

    # Avisos de Wolfe (agrupados, no dentro del loop)
    if avisos_wolfe:
        st.warning(
            f"⚠️ La búsqueda de línea de Wolfe no encontró un paso válido en "
            f"{len(avisos_wolfe)} iteración(es) (iter. {avisos_wolfe}). "
            f"Se usó α = 1e-4 como respaldo en esos casos."
        )

    # --- 4. RESULTADOS ---
    st.success("✅ Optimización finalizada exitosamente.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resultados Numéricos")
        st.write("**Punto mínimo encontrado:**")
        st.code(np.round(x, 6))
        st.write(f"**Valor de la función objetivo:** `{round(f_eval(x), 6)}`")
        st.write(f"**Gradiente final (∇f(x)):** `{np.round(grad_eval(x), 6)}`")
        st.write(f"**Número de iteraciones:** `{k}`")
        st.write(f"**Error final (||∇f(x)||):** `{format(error_actual, '.2e')}`")
        st.write(f"**Criterio de parada:** {criterio_parada}")

    with col2:
        st.subheader("Gráfico de Convergencia")
        fig, ax = plt.subplots()
        ax.plot(range(len(historial_error)), historial_error,
                marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Número de Iteraciones')
        ax.set_ylabel('Error (Norma del Gradiente)')
        ax.set_title('Evolución del Error')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        st.pyplot(fig)

    # --- VALOR AGREGADO 1: Trayectoria 2D ---
    if num_vars == 2 and len(historial_x) > 1:
        st.subheader("Trayectoria sobre curvas de nivel (2D)")
        tray = np.array(historial_x)

        margen = max(3.0, np.max(np.abs(tray - x)) * 1.5)
        x1_range = np.linspace(x[0] - margen, x[0] + margen, 300)
        x2_range = np.linspace(x[1] - margen, x[1] + margen, 300)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        try:
            Z = np.vectorize(lambda a, b: f_eval([a, b]))(X1, X2)
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            cp = ax2.contourf(X1, X2, Z, levels=40, cmap='viridis', alpha=0.75)
            ax2.contour(X1, X2, Z, levels=40, colors='white', linewidths=0.4, alpha=0.4)
            plt.colorbar(cp, ax=ax2, label='f(x)')
            ax2.plot(tray[:, 0], tray[:, 1], 'w.-', linewidth=1.5, markersize=5, label='Trayectoria')
            ax2.plot(*x0, 'go', markersize=10, label='Inicio', zorder=5)
            ax2.plot(*x,  'r*', markersize=14, label='Mínimo', zorder=5)
            ax2.set_xlabel('x1')
            ax2.set_ylabel('x2')
            ax2.set_title(f'Trayectoria del método: {metodo}')
            ax2.legend()
            st.pyplot(fig2)
        except Exception:
            st.info("No fue posible graficar la trayectoria para esta función.")

    # --- VALOR AGREGADO 2: Tabla de iteraciones ---
    if mostrar_tabla:
        st.subheader("Tabla de iteraciones")
        tabla = {
            "Iteración": list(range(1, k + 1)),
            "f(x)":      [round(v, 8) for v in historial_f[1:]],
            "||∇f(x)||": [f"{v:.4e}" for v in historial_error[1:]],
            "α (paso)":  [f"{v:.4e}" for v in historial_alpha],
        }
        st.dataframe(tabla, use_container_width=True)
