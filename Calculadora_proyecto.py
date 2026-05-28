import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import line_search

st.set_page_config(page_title="Calculadora de Optimización", layout="wide", page_icon="📉")

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    /* Fondo general y tipografía */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    /* Título principal */
    h1 { 
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }
    /* Subheaders */
    h2, h3 { color: #00d4ff !important; }
    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    /* Métricas */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #aaa;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("📉 Calculadora de Optimización No Lineal")
st.caption("Métodos de gradiente, gradiente conjugado y Newton con condiciones de Wolfe")

# --- GUÍA DE USO ---
with st.expander("📖 Guía de uso y sintaxis matemática (Haz clic para expandir/contraer)", expanded=False):
    st.markdown("""
    **Sintaxis obligatoria para la Función Objetivo:**
    * **Variables:** Utiliza siempre el formato `x1`, `x2`, `x3`, etc. (No uses `x` o `y` de forma aislada).
    * **Potencias:** Puedes usar `^` o `**` (Ejemplo: `x1^2` o `x1**2`).
    * **Multiplicación:** Usa explícitamente el asterisco `*` (Ejemplo: `2*x1`, **no** `2x1`).
    * **Exponencial (Euler):** Para $e^{x_1}$, escribe `exp(x1)` o `e^(x1)`.
    * **Trigonometría y otras:** Puedes usar `sin(x1)`, `cos(x1)`.
    * **Logaritmos:** Usa `log(x1)` para logaritmo base 10. Para otras bases, usa `log(x1, c)` donde `c` es la base (Ejemplo: `log(x1, 2)` para log base 2, `log(x1, e)` para logaritmo natural). También puedes escribir directamente `ln(x1)` para el logaritmo natural.

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
    * **Superficie 3D interactiva** *(solo para funciones de 2 variables):* Muestra la superficie de la función con la trayectoria del algoritmo. Puedes rotar, hacer zoom e interactuar con el gráfico.
    * **Carrera de algoritmos:** Ejecuta los 3 métodos simultáneamente y compara su convergencia en un solo gráfico.
    * **Tabla de iteraciones:** Detalla el valor de `f(x)`, la norma del gradiente y el paso `α` en cada iteración.
    """)

# =============================================
# CASOS DE ESTUDIO PREDISEÑADOS
# =============================================
CASOS = {
    "── Seleccionar caso ──": None,
    "🟢 Cuadrática simple (2 vars)": {
        "desc": "Función cuadrática convexa. Mínimo en el origen.",
        "nvars": 2, "func": "x1**2 + x2**2", "x0": "2.0, 2.0"
    },
    "🟡 Rosenbrock (2 vars)": {
        "desc": "Función clásica de prueba. Mínimo en (1,1), difícil de encontrar por el valle curvo.",
        "nvars": 2, "func": "100*(x2 - x1**2)**2 + (1 - x1)**2", "x0": "-1.0, 1.0"
    },
    "🔵 Himmelblau (2 vars)": {
        "desc": "Función con múltiples mínimos locales. Un mínimo en (3, 2).",
        "nvars": 2, "func": "(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2", "x0": "0.0, 0.0"
    },
    "🔴 Minimización de costo logístico (3 vars)": {
        "desc": "Modelo de costos de transporte: minimizar costo total de distribución entre 3 rutas.",
        "nvars": 3, "func": "2*x1**2 + 3*x2**2 + x3**2 - 4*x1 - 6*x2 - 2*x3 + 10", "x0": "0.0, 0.0, 0.0"
    },
    "🟣 Diseño térmico de aislación (2 vars)": {
        "desc": "Minimiza la pérdida de calor en función del espesor (x1) y conductividad (x2) del aislante.",
        "nvars": 2, "func": "(x1 - 2)**2 + 5*(x2 - 1)**2 + x1*x2", "x0": "0.5, 0.5"
    },
    "⚫ Cuadrática alta dimensión (5 vars)": {
        "desc": "Cuadrática convexa en 5 variables. Útil para comparar velocidad de convergencia entre métodos.",
        "nvars": 5, "func": "x1**2 + 2*x2**2 + 3*x3**2 + 4*x4**2 + 5*x5**2", "x0": "1.0, 1.0, 1.0, 1.0, 1.0"
    },
}

# =============================================
# SIDEBAR
# =============================================
st.sidebar.header("⚙️ Datos de Entrada")

caso_sel = st.sidebar.selectbox("📂 Cargar caso de estudio", list(CASOS.keys()))
caso = CASOS[caso_sel]

if caso:
    st.sidebar.info(caso["desc"])

# Valores por defecto según caso seleccionado
default_nvars = caso["nvars"] if caso else 2
default_func  = caso["func"]  if caso else "x1**2 + x2**2"
default_x0    = caso["x0"]    if caso else "2.0, 2.0"

st.sidebar.markdown("---")
num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=50, value=default_nvars, step=1)
metodo   = st.sidebar.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
func_str = st.sidebar.text_input("Función objetivo", default_func)
punto_partida_str = st.sidebar.text_input("Punto de partida (separado por comas)", default_x0)
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)
tol      = st.sidebar.number_input("Tolerancia de convergencia", min_value=1e-10, value=1e-5, format="%.5f", step=1e-5)

st.sidebar.markdown("**Parámetros de Wolfe**")
c1 = st.sidebar.number_input("c1 (Suficiente descenso)", min_value=0.0001, max_value=0.9999, value=1e-4, format="%.4f", step=0.0001)
c2 = st.sidebar.number_input("c2 (Curvatura)", min_value=c1, max_value=0.9999, value=0.9, format="%.4f", step=0.1000)

st.sidebar.markdown("---")
mostrar_tabla   = st.sidebar.checkbox("Mostrar tabla de iteraciones", value=False)
modo_carrera    = st.sidebar.checkbox("🏁 Carrera de algoritmos (comparar los 3 métodos)", value=False)

# =============================================
# PROCESAMIENTO MATEMÁTICO
# =============================================
try:
    func_str_procesada = func_str.replace('^', '**')
    vars_simbolicas = sp.symbols(f'x1:{num_vars + 1}')
    def log_flexible(x, base=10): return sp.log(x, base)
    f_simbolica = sp.sympify(func_str_procesada, locals={'e': sp.E, 'ln': sp.log, 'log': log_flexible})

    gradiente_simbolico  = [sp.diff(f_simbolica, var) for var in vars_simbolicas]
    hessiano_simbolico   = [[sp.diff(g, var) for var in vars_simbolicas] for g in gradiente_simbolico]

    f_num    = sp.lambdify(vars_simbolicas, f_simbolica, 'numpy')
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

# =============================================
# FUNCIÓN DE OPTIMIZACIÓN REUTILIZABLE
# =============================================
def ejecutar_metodo(metodo_nombre, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars):
    x = x0.copy()
    historial_error = [np.linalg.norm(grad_eval(x))]
    historial_f     = [f_eval(x)]
    historial_x     = [x.copy()]
    historial_alpha = []
    avisos_wolfe    = []
    p_previo = grad_previo = None
    k = 0
    criterio_parada = "Número máximo de iteraciones alcanzado"

    while k < max_iter and historial_error[-1] > tol:
        gk = grad_eval(x)

        if metodo_nombre == "Gradiente":
            pk = -gk
        elif metodo_nombre == "Gradiente Conjugado":
            if k == 0 or k % num_vars == 0:
                pk = -gk
            else:
                denom = np.dot(grad_previo, grad_previo)
                beta  = np.dot(gk, gk) / denom if denom >= 1e-12 else 0.0
                pk    = -gk + beta * p_previo
            p_previo    = pk.copy()
            grad_previo = gk.copy()
        elif metodo_nombre == "Newton":
            Hk = hess_eval(x)
            try:
                pk = np.linalg.solve(Hk, -gk)
                if np.dot(pk, gk) >= 0:
                    pk = -gk
            except np.linalg.LinAlgError:
                pk = -gk

        alpha, *_ = line_search(f_eval, grad_eval, x, pk, gfk=gk, old_fval=f_eval(x), c1=c1, c2=c2)
        if alpha is None:
            alpha = 1e-4
            avisos_wolfe.append(k + 1)

        historial_alpha.append(alpha)
        x = x + alpha * pk
        historial_error.append(np.linalg.norm(grad_eval(x)))
        historial_f.append(f_eval(x))
        historial_x.append(x.copy())
        k += 1

    if historial_error[-1] <= tol:
        criterio_parada = "Tolerancia de convergencia alcanzada"

    return {
        "x": x, "k": k, "error": historial_error[-1],
        "historial_error": historial_error, "historial_f": historial_f,
        "historial_x": historial_x, "historial_alpha": historial_alpha,
        "avisos_wolfe": avisos_wolfe, "criterio_parada": criterio_parada
    }

# =============================================
# BOTÓN Y EJECUCIÓN
# =============================================
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    ejecutar = st.button("▶️ Ejecutar Optimización")

if ejecutar:

    # ---- MODO CARRERA ----
    if modo_carrera:
        st.markdown("## 🏁 Carrera de Algoritmos")
        st.caption("Los 3 métodos corren sobre la misma función y punto de partida.")

        barra = st.progress(0, text="Ejecutando métodos...")
        metodos_carrera = ["Gradiente", "Gradiente Conjugado", "Newton"]
        colores = {"Gradiente": "#00d4ff", "Gradiente Conjugado": "#f7b731", "Newton": "#7b2ff7"}
        resultados_carrera = {}

        for i, m in enumerate(metodos_carrera):
            resultados_carrera[m] = ejecutar_metodo(m, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars)
            barra.progress((i + 1) / 3, text=f"Completado: {m}")

        # Gráfico de convergencia comparativo
        fig_carrera = go.Figure()
        for m, res in resultados_carrera.items():
            fig_carrera.add_trace(go.Scatter(
                x=list(range(len(res["historial_error"]))),
                y=res["historial_error"],
                mode="lines+markers",
                name=f"{m} ({res['k']} iter.)",
                line=dict(color=colores[m], width=2),
                marker=dict(size=4)
            ))
        fig_carrera.update_layout(
            title="Comparación de convergencia — Norma del Gradiente",
            xaxis_title="Iteraciones",
            yaxis_title="||∇f(x)||",
            yaxis_type="log",
            template="plotly_dark",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            height=420
        )
        st.plotly_chart(fig_carrera, use_container_width=True)

        # Tabla comparativa de resultados
        st.markdown("### Resumen comparativo")
        cols = st.columns(3)
        for i, (m, res) in enumerate(resultados_carrera.items()):
            with cols[i]:
                ganador = res["criterio_parada"] == "Tolerancia de convergencia alcanzada"
                st.metric(label=m, value=f"{res['k']} iteraciones",
                          delta="✅ Convergió" if ganador else "⚠️ No convergió")
                st.caption(f"f(x*) = {round(f_eval(res['x']), 6)}")
                st.caption(f"Error = {res['error']:.2e}")

        st.markdown("---")
        st.markdown("### Resultado individual por método")
        tabs_carrera = st.tabs(metodos_carrera)
        for tab, m in zip(tabs_carrera, metodos_carrera):
            res = resultados_carrera[m]
            with tab:
                c1c, c2c = st.columns(2)
                with c1c:
                    st.write("**Punto encontrado:**")
                    st.code(np.round(res["x"], 6))
                    st.write(f"**f(x):** `{round(f_eval(res['x']), 6)}`")
                    st.write(f"**∇f(x):** `{np.round(grad_eval(res['x']), 6)}`")
                    st.write(f"**Criterio:** {res['criterio_parada']}")
                with c2c:
                    if res["avisos_wolfe"]:
                        st.warning(f"Wolfe falló en {len(res['avisos_wolfe'])} iteraciones.")

    # ---- MODO NORMAL ----
    else:
        barra_progreso = st.progress(0, text="Ejecutando...")
        res = ejecutar_metodo(metodo, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars)
        barra_progreso.progress(1.0, text="¡Listo!")

        x            = res["x"]
        k            = res["k"]
        error_actual = res["error"]

        if res["avisos_wolfe"]:
            st.warning(
                f"⚠️ Wolfe no encontró paso válido en {len(res['avisos_wolfe'])} iteración(es). "
                f"Se usó α = 1e-4 como respaldo."
            )

        no_convergio = error_actual > tol and len(res["avisos_wolfe"]) >= k // 2
        if error_actual <= tol:
            st.success("✅ Optimización finalizada exitosamente.")
        elif no_convergio:
            st.error("❌ No se encontró un mínimo. Verifica que la función sea acotada inferiormente.")
        else:
            st.warning("⚠️ No convergió dentro del máximo de iteraciones. Considera aumentar las iteraciones.")

        # --- Métricas ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Iteraciones", k)
        m2.metric("f(x*)", round(f_eval(x), 6))
        m3.metric("Error final ||∇f||", f"{error_actual:.2e}")
        m4.metric("Criterio", "Tol. ✅" if error_actual <= tol else "Max. iter ⚠️")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Resultados Numéricos")
            st.write("**Punto mínimo encontrado:**" if error_actual <= tol else "**Último punto visitado:**")
            st.code(np.round(x, 6))
            st.write(f"**Gradiente final ∇f(x):** `{np.round(grad_eval(x), 6)}`")
            st.write(f"**Criterio de parada:** {res['criterio_parada']}")

        with col2:
            st.subheader("Gráfico de Convergencia")
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(len(res["historial_error"]))),
                y=res["historial_error"],
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=4),
                name="||∇f(x)||"
            ))
            fig_conv.update_layout(
                xaxis_title="Iteraciones",
                yaxis_title="Error (Norma del Gradiente)",
                yaxis_type="log",
                template="plotly_dark",
                height=350,
                margin=dict(t=20)
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # --- Superficie 3D interactiva ---
        if num_vars == 2 and len(res["historial_x"]) > 1:
            st.subheader("🌐 Superficie 3D Interactiva")
            tray = np.array(res["historial_x"])
            margen = max(3.0, np.max(np.abs(tray - x)) * 1.5)

            x1_r = np.linspace(x[0] - margen, x[0] + margen, 120)
            x2_r = np.linspace(x[1] - margen, x[1] + margen, 120)
            X1, X2 = np.meshgrid(x1_r, x2_r)
            try:
                Z = np.vectorize(lambda a, b: f_eval([a, b]))(X1, X2)

                z_tray = np.array([f_eval(p) for p in tray])

                fig3d = go.Figure()
                fig3d.add_trace(go.Surface(
                    x=X1, y=X2, z=Z,
                    colorscale="Viridis", opacity=0.85,
                    showscale=True,
                    contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True))
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=tray[:, 0], y=tray[:, 1], z=z_tray,
                    mode="lines+markers",
                    line=dict(color="white", width=4),
                    marker=dict(size=3, color="white"),
                    name="Trayectoria"
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=[x0[0]], y=[x0[1]], z=[f_eval(x0)],
                    mode="markers", marker=dict(size=10, color="lime"),
                    name="Inicio"
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=[x[0]], y=[x[1]], z=[f_eval(x)],
                    mode="markers", marker=dict(size=10, color="red", symbol="diamond"),
                    name="Mínimo"
                ))
                fig3d.update_layout(
                    scene=dict(
                        xaxis_title="x1", yaxis_title="x2", zaxis_title="f(x)",
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    template="plotly_dark",
                    height=550,
                    legend=dict(bgcolor="rgba(0,0,0,0.4)")
                )
                st.plotly_chart(fig3d, use_container_width=True)
                st.caption("💡 Puedes rotar, hacer zoom y explorar la superficie con el mouse.")
            except Exception:
                st.info("No fue posible graficar la superficie 3D para esta función.")

        # --- Tabla de iteraciones ---
        if mostrar_tabla:
            st.subheader("Tabla de iteraciones")
            tabla = {
                "Iteración":  list(range(1, k + 1)),
                "f(x)":       [round(v, 8) for v in res["historial_f"][1:]],
                "||∇f(x)||":  [f"{v:.4e}" for v in res["historial_error"][1:]],
                "α (paso)":   [f"{v:.4e}" for v in res["historial_alpha"]],
            }
            st.dataframe(tabla, use_container_width=True)
