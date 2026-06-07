import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.optimize import line_search
import io
import csv

st.set_page_config(page_title="Calculadora de Optimización", layout="wide", page_icon="📉")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e2a2e 0%, #223030 50%, #1e2828 100%);
        color: #ffffff;
    }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.12);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    h1 {
        background: linear-gradient(90deg, #6dd5ed, #56c596);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }
    h2, h3 { color: #6dd5ed !important; }
    .stButton > button {
        background: linear-gradient(90deg, #6dd5ed, #56c596);
        color: #1a2a2a;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(109,213,237,0.35);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] { color: #ffffff; font-weight: 600; }
    .stTabs [aria-selected="true"] {
        color: #6dd5ed !important;
        border-bottom: 2px solid #6dd5ed;
    }
</style>
""", unsafe_allow_html=True)

st.title("📉 Calculadora de Optimización No Lineal")
st.caption("Métodos de gradiente, gradiente conjugado y Newton con condiciones de Wolfe")

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
    **Gráficos y funciones disponibles:**
    * **Convergencia:** Norma del gradiente por iteración en escala logarítmica.
    * **Superficie 3D interactiva** *(solo funciones de 2 variables, modo normal):* Superficie de la función con la trayectoria del algoritmo. Puedes rotar, hacer zoom e interactuar con el gráfico.
    * **Carrera de algoritmos:** Ejecuta los 3 métodos simultáneamente y compara su convergencia. Incluye ranking automático y análisis del resultado.
    * **Tabla de iteraciones:** Valor de `f(x)`, norma del gradiente y paso `α` en cada iteración, con descarga en CSV.
    """)

CASOS = {
    "── Seleccionar caso ──": None,
    "🟢 Cuadrática simple (2 vars)": {
        "desc": "Función cuadrática convexa. Mínimo en el origen.",
        "nvars": 2, "func": "x1**2 + x2**2", "x0": "2.0, 2.0"
    },
    "🟡 Rosenbrock (2 vars)": {
        "desc": "Función clásica de prueba. Mínimo en (1,1), difícil de encontrar por el valle curvo. Se recomienda aumentar el número de iteraciones a 500 para ver convergencia completa.",
        "nvars": 2, "func": "100*(x2 - x1**2)**2 + (1 - x1)**2", "x0": "-2.0, 2.0"
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

# --- SIDEBAR ---
st.sidebar.header("⚙️ Datos de Entrada")

# Session state para sincronizar casos con los widgets
if "caso_anterior" not in st.session_state:
    st.session_state.caso_anterior = "── Seleccionar caso ──"
if "func_str" not in st.session_state:
    st.session_state.func_str = "x1**2 + x2**2"
if "x0_str" not in st.session_state:
    st.session_state.x0_str = "2.0, 2.0"
if "nvars" not in st.session_state:
    st.session_state.nvars = 2

caso_sel = st.sidebar.selectbox("📂 Cargar caso de estudio", list(CASOS.keys()))
caso = CASOS[caso_sel]

if caso_sel != st.session_state.caso_anterior:
    st.session_state.caso_anterior = caso_sel
    if caso:
        st.session_state.func_str = caso["func"]
        st.session_state.x0_str   = caso["x0"]
        st.session_state.nvars    = caso["nvars"]

if caso:
    st.sidebar.info(caso["desc"])

st.sidebar.markdown("---")
num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=50,
                                    value=st.session_state.nvars, step=1)
metodo   = st.sidebar.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
func_str = st.sidebar.text_input("Función objetivo", value=st.session_state.func_str)
punto_partida_str = st.sidebar.text_input("Punto de partida (separado por comas)", value=st.session_state.x0_str)
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)
tol      = st.sidebar.number_input("Tolerancia de convergencia", min_value=1e-10, value=1e-5, format="%.5f", step=1e-5)

st.sidebar.markdown("**Parámetros de Wolfe**")
c1 = st.sidebar.number_input("c1 (Suficiente descenso)", min_value=0.0001, max_value=0.9999, value=1e-4, format="%.4f", step=0.0001)
c2 = st.sidebar.number_input("c2 (Curvatura)", min_value=c1, max_value=0.9999, value=0.9, format="%.4f", step=0.1000)

st.sidebar.markdown("---")
mostrar_tabla = st.sidebar.checkbox("Mostrar tabla de iteraciones", value=False)
modo_carrera  = st.sidebar.checkbox("🏁 Carrera de algoritmos (comparar los 3 métodos)", value=False)

# --- PROCESAMIENTO MATEMÁTICO ---
try:
    func_str_procesada = func_str.replace('^', '**')
    vars_simbolicas = sp.symbols(f'x1:{num_vars + 1}')
    def log_flexible(x, base=10): return sp.log(x, base)
    f_simbolica = sp.sympify(func_str_procesada, locals={'e': sp.E, 'ln': sp.log, 'log': log_flexible})

    gradiente_simbolico = [sp.diff(f_simbolica, var) for var in vars_simbolicas]
    hessiano_simbolico  = [[sp.diff(g, var) for var in vars_simbolicas] for g in gradiente_simbolico]

    f_num    = sp.lambdify(vars_simbolicas, f_simbolica, 'numpy')
    grad_num = sp.lambdify(vars_simbolicas, gradiente_simbolico, 'numpy')
    hess_num = sp.lambdify(vars_simbolicas, hessiano_simbolico, 'numpy')

    def f_eval(x):
        val = float(f_num(*x))
        if not np.isfinite(val):
            raise ValueError(f"f(x) no es finito en x={x}: {val}")
        return val

    def grad_eval(x):
        g = np.array(grad_num(*x), dtype=float).flatten()
        if not np.all(np.isfinite(g)):
            raise ValueError(f"Gradiente no finito en x={x}: {g}")
        return g

    def hess_eval(x):
        return np.array(hess_num(*x), dtype=float)

    x0 = np.array([float(v.strip()) for v in punto_partida_str.split(',')])
    if len(x0) != num_vars:
        st.error(f"⚠️ El punto de partida debe tener exactamente {num_vars} componentes separados por comas.")
        st.stop()

except Exception as e:
    st.error(f"⚠️ Error en la sintaxis matemática. Por favor, revisa la guía de uso. Detalle técnico: {e}")
    st.stop()

# --- ALGORITMO DE OPTIMIZACIÓN ---
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
            # Fletcher-Reeves con reinicio cada n iteraciones
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
                # Si no es dirección de descenso, usar gradiente negativo
                if np.dot(pk, gk) >= 0:
                    pk = -gk
            except np.linalg.LinAlgError:
                pk = -gk

        fval_actual = f_eval(x)
        alpha, *_ = line_search(f_eval, grad_eval, x, pk, gfk=gk, old_fval=fval_actual, c1=c1, c2=c2)
        if alpha is None:
            alpha = 1e-4
            avisos_wolfe.append(k + 1)

        historial_alpha.append(alpha)
        x_nuevo = x + alpha * pk
        try:
            error_nuevo = np.linalg.norm(grad_eval(x_nuevo))
            f_nuevo     = f_eval(x_nuevo)
        except ValueError:
            criterio_parada = "Divergencia detectada: la función no es finita en el nuevo punto"
            break

        x = x_nuevo
        historial_error.append(error_nuevo)
        historial_f.append(f_nuevo)
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


def generar_csv(res, k):
    """Genera un CSV con el historial de iteraciones."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Iteracion", "f(x)", "||grad_f(x)||", "alpha"])
    for i in range(k):
        writer.writerow([
            i + 1,
            round(res["historial_f"][i + 1], 8),
            f"{res['historial_error'][i + 1]:.6e}",
            f"{res['historial_alpha'][i]:.6e}",
        ])
    return output.getvalue().encode("utf-8")


def grafico_3d(res, x0, x, f_eval):
    """Genera y muestra la superficie 3D con la trayectoria del método."""
    tray   = np.array(res["historial_x"])
    margen = min(max(3.0, np.max(np.abs(tray - x)) * 1.5), 20.0)

    x1_r = np.linspace(x[0] - margen, x[0] + margen, 120)
    x2_r = np.linspace(x[1] - margen, x[1] + margen, 120)
    X1, X2 = np.meshgrid(x1_r, x2_r)

    try:
        Z      = np.vectorize(lambda a, b: f_eval([a, b]))(X1, X2)
        z_tray = np.array([f_eval(p) for p in tray])

        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            x=X1, y=X2, z=Z, colorscale="Viridis", opacity=0.85,
            showscale=True,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True))
        ))
        fig3d.add_trace(go.Scatter3d(
            x=tray[:, 0], y=tray[:, 1], z=z_tray,
            mode="lines+markers", line=dict(color="white", width=4),
            marker=dict(size=3, color="white"), name="Trayectoria"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[x0[0]], y=[x0[1]], z=[f_eval(x0)],
            mode="markers", marker=dict(size=10, color="lime"), name="Inicio"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[x[0]], y=[x[1]], z=[f_eval(x)],
            mode="markers", marker=dict(size=10, color="red", symbol="diamond"), name="Mínimo"
        ))
        fig3d.update_layout(
            scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="f(x)", bgcolor="rgba(0,0,0,0)"),
            template="plotly_dark", height=550, legend=dict(bgcolor="rgba(0,0,0,0.4)")
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("💡 Puedes rotar, hacer zoom y explorar la superficie con el mouse.")
    except Exception:
        st.info("No fue posible graficar la superficie 3D para esta función.")


def ranking_carrera(resultados_carrera, f_eval):
    """Genera tabla de ranking comparativo entre los 3 métodos."""
    metodos = list(resultados_carrera.keys())

    # Determinar ganador por iteraciones (entre los que convergieron)
    convergidos = [m for m in metodos if resultados_carrera[m]["criterio_parada"] == "Tolerancia de convergencia alcanzada"]
    if convergidos:
        ganador_iter  = min(convergidos, key=lambda m: resultados_carrera[m]["k"])
        ganador_error = min(convergidos, key=lambda m: resultados_carrera[m]["error"])
    else:
        ganador_iter = ganador_error = None

    st.markdown("### 🏆 Ranking comparativo")
    cols = st.columns(3)
    for i, m in enumerate(metodos):
        res     = resultados_carrera[m]
        convergio = res["criterio_parada"] == "Tolerancia de convergencia alcanzada"
        with cols[i]:
            tags = []
            if m == ganador_iter:  tags.append("🥇 Más rápido")
            if m == ganador_error: tags.append("🎯 Mejor precisión")
            if not convergio:      tags.append("⚠️ No convergió")

            st.metric(label=m, value=f"{res['k']} iter.", delta=" | ".join(tags) if tags else "✅ Convergió")
            st.caption(f"f(x*) = {round(f_eval(res['x']), 8)}")
            st.caption(f"Error final = {res['error']:.2e}")

    # Análisis textual automático
    st.markdown("#### 📋 Análisis de resultados")
    lineas = []
    if ganador_iter:
        lineas.append(f"- **{ganador_iter}** fue el método más rápido con {resultados_carrera[ganador_iter]['k']} iteraciones.")
    if ganador_error and ganador_error != ganador_iter:
        lineas.append(f"- **{ganador_error}** obtuvo el menor error final ({resultados_carrera[ganador_error]['error']:.2e}).")
    no_conv = [m for m in metodos if m not in convergidos]
    if no_conv:
        lineas.append(f"- **{', '.join(no_conv)}** no alcanzó la tolerancia. Considera aumentar el número de iteraciones.")
    if not lineas:
        lineas.append("- Los 3 métodos convergieron. Compara las iteraciones y el error final para elegir el más eficiente.")
    st.markdown("\n".join(lineas))


# --- EJECUCIÓN ---
ejecutar = st.button("▶️ Ejecutar Optimización")

if ejecutar:

    # MODO CARRERA
    if modo_carrera:
        st.markdown("## 🏁 Carrera de Algoritmos")
        st.caption("Los 3 métodos corren sobre la misma función y punto de partida.")

        barra = st.progress(0, text="Ejecutando métodos...")
        metodos_carrera = ["Gradiente", "Gradiente Conjugado", "Newton"]
        colores = {"Gradiente": "#6dd5ed", "Gradiente Conjugado": "#f7b731", "Newton": "#56c596"}
        resultados_carrera = {}

        for i, m in enumerate(metodos_carrera):
            resultados_carrera[m] = ejecutar_metodo(m, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars)
            barra.progress((i + 1) / 3, text=f"Completado: {m}")

        # Gráfico comparativo de convergencia
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
            xaxis_title="Iteraciones", yaxis_title="||∇f(x)||",
            yaxis_type="log", template="plotly_dark",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"), height=420
        )
        st.plotly_chart(fig_carrera, use_container_width=True)

        ranking_carrera(resultados_carrera, f_eval)

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
                    if mostrar_tabla and res["k"] > 0:
                        st.download_button(
                            label=f"⬇️ Descargar CSV — {m}",
                            data=generar_csv(res, res["k"]),
                            file_name=f"historial_{m.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )

    # MODO NORMAL
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

        no_convergio = error_actual > tol and k > 0 and len(res["avisos_wolfe"]) >= k // 2
        if error_actual <= tol:
            st.success("✅ Optimización finalizada exitosamente.")
        elif no_convergio:
            st.error("❌ No se encontró un mínimo. Verifica que la función sea acotada inferiormente.")
        else:
            st.warning("⚠️ No convergió dentro del máximo de iteraciones. Considera aumentar las iteraciones.")

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
                line=dict(color="#6dd5ed", width=2),
                marker=dict(size=4), name="||∇f(x)||"
            ))
            fig_conv.update_layout(
                xaxis_title="Iteraciones", yaxis_title="Error (Norma del Gradiente)",
                yaxis_type="log", template="plotly_dark",
                height=350, margin=dict(t=20)
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # Superficie 3D (solo modo normal, 2 variables)
        if num_vars == 2 and len(res["historial_x"]) > 1:
            st.subheader("🌐 Superficie 3D Interactiva")
            grafico_3d(res, x0, x, f_eval)

        # Tabla de iteraciones con descarga CSV
        if mostrar_tabla and k > 0:
            st.subheader("Tabla de iteraciones")
            tabla = {
                "Iteración":  list(range(1, k + 1)),
                "f(x)":       [round(v, 8) for v in res["historial_f"][1:]],
                "||∇f(x)||":  [f"{v:.4e}" for v in res["historial_error"][1:]],
                "α (paso)":   [f"{v:.4e}" for v in res["historial_alpha"]],
            }
            st.dataframe(tabla, use_container_width=True)
            st.download_button(
                label="⬇️ Descargar tabla como CSV",
                data=generar_csv(res, k),
                file_name="historial_iteraciones.csv",
                mime="text/csv"
            )
