import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.optimize import line_search
import io
import csv
from datetime import datetime

st.set_page_config(
    page_title="Calculadora de Optimización",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #1e2a2e 0%, #223030 50%, #1e2828 100%);
        color: #ffffff;
    }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.12);
    }
    h1 {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, #6dd5ed, #56c596);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #6dd5ed !important;
        font-weight: 700;
    }
    div[data-testid="stMarkdownContainer"] hr {
        border-color: rgba(255,255,255,0.2);
        margin: 1.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6dd5ed, #56c596);
        color: #1a2a2a;
        border: none;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.1rem;
        padding: 0.8rem 2.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        opacity: 0.95;
    }
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(109,213,237,0.35);
        border-radius: 16px;
        padding: 1.5rem;
    }
    [data-testid="stMetricValue"] > div {
        color: #6dd5ed !important;
        font-family: 'Roboto Mono', monospace;
        font-size: 2.2rem !important;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] { color: #e0e0e0; font-weight: 600; }
    .stTabs [aria-selected="true"] {
        color: #6dd5ed !important;
        border-bottom: 2px solid #6dd5ed;
    }
    code {
        font-family: 'Roboto Mono', monospace;
        background: rgba(255,255,255,0.1) !important;
        color: #56c596 !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Calculadora de Optimización No Lineal")
st.caption("Métodos de gradiente, gradiente conjugado y Newton con condiciones de Wolfe")

with st.expander("📖 Guía de uso y sintaxis matemática (Haz clic para expandir/contraer)", expanded=False):
    st.markdown("""
    **Sintaxis obligatoria para la Función Objetivo:**
    * **Variables:** Utiliza siempre el formato `x1`, `x2`, `x3`, etc.
    * **Potencias:** Puedes usar `^` o `**` (Ejemplo: `x1^2` o `x1**2`).
    * **Multiplicación:** Usa explícitamente el asterisco `*` (Ejemplo: `2*x1`).
    * **Exponencial y Logaritmos:** Usa `exp(x1)`, `ln(x1)` para logaritmo natural, `log(x1)` para base 10, `log(x1, c)` para base `c`.
    """)

# --- CASOS DE ESTUDIO ---
CASOS = {
    "── Seleccionar caso de ejemplo ──": None,
    "🟢 Cuadrática simple (2 vars)": {
        "desc": "Minimización de una parábola convexa. El mínimo global está exactamente en el origen (0,0).",
        "nvars": 2, "func": "x1**2 + x2**2", "x0": "2.0, 2.0"
    },
    "🟡 Valle de Rosenbrock (2 vars)": {
        "desc": "El 'Valle de la Banana'. Mínimo global en (1,1). Difícil convergencia. Se recomienda aumentar las iteraciones a 500.",
        "nvars": 2, "func": "100*(x2 - x1**2)**2 + (1 - x1)**2", "x0": "-2.0, 2.0"
    },
    "🔵 Función Himmelblau (2 vars)": {
        "desc": "Función con 4 mínimos locales. Prueba de mínimos cercanos.",
        "nvars": 2, "func": "(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2", "x0": "0.0, 0.0"
    },
    "🔴 Minimización de Costo Logístico (3 vars)": {
        "desc": "Minimizar el costo total de distribución entre 3 rutas de transporte.",
        "nvars": 3, "func": "2*x1**2 + 3*x2**2 + x3**2 - 4*x1 - 6*x2 - 2*x3 + 10", "x0": "1.0, 1.0, 1.0"
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
st.sidebar.markdown("### 📂 Biblioteca de Modelos")
caso_sel = st.sidebar.selectbox("Cargar ejemplo prediseñado", list(CASOS.keys()))
caso = CASOS[caso_sel]

if "caso_anterior" not in st.session_state:
    st.session_state.caso_anterior = "── Seleccionar caso de ejemplo ──"
if "func_str" not in st.session_state:
    st.session_state.func_str = "x1**2 + x2**2"
if "x0_str" not in st.session_state:
    st.session_state.x0_str = "2.0, 2.0"
if "nvars" not in st.session_state:
    st.session_state.nvars = 2

# Solo actualizar cuando cambia el caso, no en cada rerun
if caso_sel != st.session_state.caso_anterior:
    st.session_state.caso_anterior = caso_sel
    if caso:
        st.session_state.func_str = caso["func"]
        st.session_state.x0_str   = caso["x0"]
        st.session_state.nvars    = caso["nvars"]

if caso:
    st.sidebar.info(caso["desc"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Configuración del Problema")
num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=50, value=st.session_state.nvars, step=1)
func_str = st.sidebar.text_input("Función objetivo f(x)", value=st.session_state.func_str)
punto_partida_str = st.sidebar.text_input("Punto de partida x₀ (separado por comas)", value=st.session_state.x0_str)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Parámetros del Algoritmo")
metodo   = st.sidebar.selectbox("Método principal", ["Gradiente", "Gradiente Conjugado", "Newton"])
max_iter = st.sidebar.number_input("Máximo iteraciones", min_value=1, value=100)
tol      = st.sidebar.number_input("Tolerancia (||∇f||)", min_value=1e-10, value=1e-5, format="%.5f", step=1e-5)

with st.sidebar.expander("Condiciones de Búsqueda de Wolfe", expanded=False):
    c1 = st.number_input("c1 (Descenso)", min_value=0.0001, max_value=0.9999, value=1e-4, format="%.4f")
    c2 = st.number_input("c2 (Curvatura)", min_value=c1, max_value=0.9999, value=0.9, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Visualización")
mostrar_tabla = st.sidebar.checkbox("Mostrar tabla de iteraciones", value=False)
modo_carrera  = st.sidebar.checkbox("🏁 Modo Carrera (Comparar los 3 métodos)", value=False)

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
        if not np.isfinite(val): raise ValueError("f(x) no finita")
        return val

    def grad_eval(x):
        g = np.array(grad_num(*x), dtype=float).flatten()
        if not np.all(np.isfinite(g)): raise ValueError("∇f no finito")
        return g

    def hess_eval(x): return np.array(hess_num(*x), dtype=float)

    x0 = np.array([float(v.strip()) for v in punto_partida_str.split(',')])
    if len(x0) != num_vars:
        st.error(f"⚠️ x₀ debe tener {num_vars} componentes.")
        st.stop()

except Exception as e:
    st.error(f"⚠️ Error en sintaxis matemática. Revisa la guía. Detalle: {e}")
    st.stop()

# --- ALGORITMO ---
def ejecutar_metodo(metodo_nombre, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars):
    x = x0.copy()
    f0 = f_eval(x)
    historial_error = [np.linalg.norm(grad_eval(x))]
    historial_f     = [f0]
    historial_x     = [x.copy()]
    historial_alpha = []
    avisos_wolfe    = []
    p_previo = grad_previo = None
    k = 0
    criterio_parada = "Max iteraciones"

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
            p_previo = pk.copy()
            grad_previo = gk.copy()
        elif metodo_nombre == "Newton":
            Hk = hess_eval(x)
            try:
                pk = np.linalg.solve(Hk, -gk)
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
            criterio_parada = "Divergencia"
            break

        x = x_nuevo
        historial_error.append(error_nuevo)
        historial_f.append(f_nuevo)
        historial_x.append(x.copy())
        k += 1

    if historial_error[-1] <= tol:
        criterio_parada = "Tol. alcanzada"

    return {
        "x": x, "k": k, "error": historial_error[-1], "f0": f0,
        "historial_error": historial_error, "historial_f": historial_f,
        "historial_x": historial_x, "historial_alpha": historial_alpha,
        "avisos_wolfe": avisos_wolfe, "criterio_parada": criterio_parada
    }

def generar_csv(res, k):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Iteracion", "f(x)", "||grad_f(x)||", "alpha"])
    for i in range(k):
        writer.writerow([i+1, round(res["historial_f"][i+1], 8),
                         f"{res['historial_error'][i+1]:.6e}",
                         f"{res['historial_alpha'][i]:.6e}"])
    return output.getvalue().encode("utf-8")

# --- EJECUCIÓN ---
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
with col_btn2:
    ejecutar = st.button("▶️ INICIAR OPTIMIZACIÓN", use_container_width=True)

if ejecutar:

    # MODO CARRERA
    if modo_carrera:
        st.markdown("## 🏁 Carrera de Algoritmos")
        barra = st.progress(0, text="Corriendo simulaciones...")
        metodos_carrera = ["Gradiente", "Gradiente Conjugado", "Newton"]
        colores = {"Gradiente": "#6dd5ed", "Gradiente Conjugado": "#f7b731", "Newton": "#56c596"}
        resultados = {}

        for i, m in enumerate(metodos_carrera):
            resultados[m] = ejecutar_metodo(m, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars)
            barra.progress((i + 1) / 3, text=f"Completado: {m}")

        fig_carrera = go.Figure()
        for m, res in resultados.items():
            fig_carrera.add_trace(go.Scatter(
                x=list(range(len(res["historial_error"]))),
                y=res["historial_error"],
                mode="lines+markers",
                name=f"{m} ({res['k']} iter.)",
                line=dict(color=colores[m], width=2.5),
                marker=dict(size=4)
            ))
        fig_carrera.update_layout(
            title="Comparación de Velocidad de Convergencia",
            xaxis_title="Iteraciones", yaxis_title="||∇f(x)|| (Escala Log)",
            yaxis_type="log", template="plotly_dark", height=450,
            legend=dict(bgcolor="rgba(0,0,0,0.5)")
        )
        st.plotly_chart(fig_carrera, use_container_width=True)

        # Ranking con más rápido Y mejor precisión separados
        st.markdown("### 🏆 Ranking de Eficiencia")
        metodos_convergidos = [m for m in metodos_carrera if resultados[m]["criterio_parada"] == "Tol. alcanzada"]
        ganador_vel   = min(metodos_convergidos, key=lambda m: resultados[m]["k"])           if metodos_convergidos else None
        ganador_prec  = min(metodos_convergidos, key=lambda m: resultados[m]["error"])       if metodos_convergidos else None

        cols_c = st.columns(3)
        for i, m in enumerate(metodos_carrera):
            res = resultados[m]
            convergio = m in metodos_convergidos
            with cols_c[i]:
                tags = []
                if m == ganador_vel:  tags.append("🥇 Más rápido")
                if m == ganador_prec: tags.append("🎯 Mejor precisión")
                if not convergio:     tags.append("⚠️ No convergió")
                if convergio and not tags:
                    tags.append("✅ Convergió")

                delta_text = " | ".join(tags)
                st.metric(label=m, value=f"{res['k']} iteraciones", delta=delta_text)
                c_fx, c_err = st.columns(2)
                c_fx.caption(f"f(x*): `{round(f_eval(res['x']), 6)}`")
                c_err.caption(f"Error: `{res['error']:.1e}`")

        # Análisis textual
        st.markdown("#### 📋 Análisis de resultados")
        lineas = []
        if ganador_vel:
            lineas.append(f"- **{ganador_vel}** fue el método más rápido con {resultados[ganador_vel]['k']} iteraciones.")
        if ganador_prec and ganador_prec != ganador_vel:
            lineas.append(f"- **{ganador_prec}** obtuvo el menor error final ({resultados[ganador_prec]['error']:.2e}).")
        no_conv = [m for m in metodos_carrera if m not in metodos_convergidos]
        if no_conv:
            lineas.append(f"- **{', '.join(no_conv)}** no alcanzó la tolerancia. Considera aumentar el número de iteraciones.")
        if not lineas:
            lineas.append("- Los 3 métodos convergieron. Compara las iteraciones y el error final para elegir el más eficiente.")
        st.markdown("\n".join(lineas))

        st.markdown("---")
        tabs = st.tabs(metodos_carrera)
        for tab, m in zip(tabs, metodos_carrera):
            res = resultados[m]
            with tab:
                c1t, c2t = st.columns([0.6, 0.4])
                c1t.write("**Punto Mínimo Encontrado x*:**")
                c1t.code(np.round(res["x"], 6))
                c2t.write(f"**Valor Final f(x*):** `{round(f_eval(res['x']), 6)}`")
                c2t.write(f"**Criterio de Parada:** {res['criterio_parada']}")
                if mostrar_tabla and res["k"] > 0:
                    c2t.download_button(
                        f"⬇️ Descargar Historial CSV ({m})",
                        data=generar_csv(res, res["k"]),
                        file_name=f"opt_{m.lower().replace(' ','_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # MODO NORMAL
    else:
        with st.spinner(f"Ejecutando {metodo}..."):
            res = ejecutar_metodo(metodo, x0, f_eval, grad_eval, hess_eval, max_iter, tol, c1, c2, num_vars)

        st.markdown("### 📊 Resumen de Resultados")
        mcols = st.columns(4)
        mcols[0].metric("Iteraciones Realizadas", res["k"])
        mcols[1].metric("Valor Mínimo f(x*)", round(f_eval(res["x"]), 6), f"Desde {round(res['f0'], 2)}")
        mcols[2].metric("Error Final ||∇f||", f"{res['error']:.2e}")
        mcols[3].metric("Estado Final", "✅ Éxito" if res["error"] <= tol else "⚠️ Incompleto", res["criterio_parada"])

        st.markdown("---")

        c_num, c_gra = st.columns([0.4, 0.6])
        with c_num:
            st.subheader("🪐 Resultados Numéricos")
            st.write("**Punto x* Encontrado:**")
            st.code(np.round(res["x"], 6))
            st.write("**Gradiente Final ∇f(x*):**")
            st.code(np.round(grad_eval(res["x"]), 6))

            # Bitácora para IA
            with st.expander("📋 Generar Bitácora para IA", expanded=False):
                st.markdown("Copia este reporte para analizarlo con IA (ChatGPT, Claude, Gemini):")
                fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
                reporte_ia = f"""BITÁCORA DE OPTIMIZACIÓN NO LINEAL
Fecha: {fecha_actual}

1. DEFINICIÓN DEL PROBLEMA
- Función objetivo f(x): {func_str}
- Número de variables: {num_vars}
- Punto de partida x₀: {np.round(x0, 4).tolist()}

2. CONFIGURACIÓN DEL ALGORITMO
- Método: {metodo}
- Tolerancia ||∇f(x)||: {tol}
- Condiciones de Wolfe: c1={c1}, c2={c2}
- Máximo de iteraciones: {max_iter}

3. RESULTADOS
- Punto mínimo x*: {np.round(res['x'], 6).tolist()}
- Valor mínimo f(x*): {round(f_eval(res['x']), 6)}
- Norma del gradiente final: {res['error']:.2e}
- Iteraciones realizadas: {res['k']}
- Criterio de parada: {res['criterio_parada']}"""
                st.code(reporte_ia, language="markdown")
                st.caption("☝️ Usa el botón de copiado en la esquina superior derecha del cuadro.")

            if mostrar_tabla and res["k"] > 0:
                st.download_button(
                    "⬇️ Descargar Tabla de Iteraciones (CSV)",
                    data=generar_csv(res, res["k"]),
                    file_name="opt_resultados.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with c_gra:
            st.subheader("📈 Curva de Convergencia")
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(len(res["historial_error"]))),
                y=res["historial_error"],
                mode="lines+markers",
                line=dict(color="#6dd5ed", width=3),
                marker=dict(size=5),
                name="||∇f(x)||"
            ))
            fig_conv.update_layout(
                xaxis_title="Iteraciones", yaxis_title="Error (Norma Gradiente)",
                yaxis_type="log", template="plotly_dark",
                height=380, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # Superficie 3D
        if num_vars == 2 and res["k"] > 0:
            st.markdown("---")
            st.subheader("🌐 Visualización 3D de la Trayectoria")
            tray    = np.array(res["historial_x"])
            x_final = res["x"]
            margen  = min(max(3.0, np.max(np.linalg.norm(tray - x_final, axis=1)) * 1.3), 15.0)

            X1, X2 = np.meshgrid(
                np.linspace(x_final[0] - margen, x_final[0] + margen, 130),
                np.linspace(x_final[1] - margen, x_final[1] + margen, 130)
            )
            try:
                Z      = np.vectorize(lambda a, b: f_eval([a, b]))(X1, X2)
                z_tray = np.array([f_eval(p) for p in tray])

                fig3d = go.Figure()
                fig3d.add_trace(go.Surface(
                    x=X1, y=X2, z=Z, colorscale="Viridis", opacity=0.8,
                    showscale=False,
                    contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=tray[:, 0], y=tray[:, 1], z=z_tray,
                    mode="lines+markers",
                    line=dict(color="#ffffff", width=4),
                    marker=dict(size=4, color="#ffffff"),
                    name="Trayectoria"
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=[x0[0]], y=[x0[1]], z=[f_eval(x0)],
                    mode="markers",
                    marker=dict(size=12, color="#00ff00",
                                line=dict(width=2, color="#000000")),
                    name="Inicio x₀"
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=[x_final[0]], y=[x_final[1]], z=[f_eval(x_final)],
                    mode="markers",
                    marker=dict(size=14, color="#ff4444", symbol="diamond",
                                line=dict(width=2, color="#ffffff")),
                    name="Mínimo x*"
                ))
                fig3d.update_layout(
                    template="plotly_dark", height=650,
                    margin=dict(t=0, b=0, l=0, r=0),
                    scene=dict(
                        xaxis_title="x1", yaxis_title="x2", zaxis_title="f(x)",
                        bgcolor="rgba(0,0,0,0)",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    legend=dict(
                        bgcolor="rgba(0,0,0,0.6)",
                        yanchor="top", y=0.95,
                        xanchor="left", x=0.05,
                        font=dict(size=13),
                        itemsizing="constant"
                    )
                )
                st.plotly_chart(fig3d, use_container_width=True)
                st.caption("💡 Puedes rotar, hacer zoom y explorar la superficie con el mouse.")
            except Exception:
                st.info("No se pudo generar el gráfico 3D para esta función.")
