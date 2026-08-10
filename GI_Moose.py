import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import os
import math

st.set_page_config(
    page_title="Growth Index Engine — Producción",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# --- ENCABEZADO CON LOGOS (WPP Izquierda, Título Centro, Moose Derecha) ---
col_logo1, col_title, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    if os.path.exists("Logo_WPP.png"):
        st.image("Logo_WPP.png", width=130)
    else:
        st.markdown("**[Logo WPP]**")

with col_title:
    st.title("🌐 Growth Index (GI)")

with col_logo2:
    if os.path.exists("Moose_logo.png"):
        st.image("Moose_logo.png", width=130)
    else:
        st.markdown("**[Logo Moose]**")

MEDIA_PARAMS = {
    'Digital': {'name': 'Digital', 'color': '#0668E1'},
    'Open Tv': {'name': 'Open Tv', 'color': '#F59E0B'}
}
REVENUE_PER_GI = 600000

if 'opt_spends' not in st.session_state or not all(k in st.session_state.opt_spends for k in MEDIA_PARAMS):
    st.session_state.opt_spends = {
        'Digital': 350000, 'Open Tv': 750000
    }
if 'opt_weeks' not in st.session_state or not all(k in st.session_state.opt_weeks for k in MEDIA_PARAMS):
    st.session_state.opt_weeks = {
        'Digital': 12, 'Open Tv': 8
    }

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Growth Index Builder", "📈 MiA - Medición del Impacto de Medios", "⚡ Budget Allocator"])

with tab1:
    st.markdown("### 📊 Growth Index Builder")
    st.markdown("Sube tu archivo CSV histórico para calibrar el modelo econométrico de Variable Latente.")

    with st.container(border=True):
        st.subheader("📁 Carga de Datos Históricos (CSV)")
        uploaded_file = st.file_uploader("Selecciona o arrastra tu archivo CSV para Growth Index", type=["csv"], key="gi_csv_upload")
        model_strategy = st.radio(
            "Estrategia Econométrica de Asignación:",
            ["Auto-detectar (Basado en columnas)", "Tipo Bluey (Menciones+Engagement 32%)", "Tipo Little Live Pets (Búsqueda 30%, Precio 28%, Menciones 13%)"],
            horizontal=True
        )

    if uploaded_file is None:
        st.warning("⚠️ Por favor, carga tu archivo CSV histórico en el bloque superior para inicializar el motor del Growth Index y visualizar los paneles de esta pestaña.")
    else:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Archivo **{uploaded_file.name}** cargado e integrado correctamente ({len(raw_df)} registros detectados).")
            
            if "Tipo Bluey" in model_strategy:
                model_mode = "bluey"
            elif "Tipo Little Live Pets" in model_strategy:
                model_mode = "llp"
            else:
                model_mode = "auto"

            @st.cache_data
            def fit_growth_index_model(dataframe, mode="auto"):
                df = dataframe.copy()
                column_mapping = {
                    'IdxPrecio': 'PrecioPromedio',
                    'Sales_Units': 'SalesUnides',
                    'SalesUnits': 'SalesUnides',
                    'Search': 'Searching'
                }
                df.rename(columns=column_mapping, inplace=True)

                if "SalesUnides" not in df.columns:
                    st.error("El CSV debe contener una columna de ventas ('SalesUnides', 'Sales_Units' or 'SalesUnits').")
                    st.stop()

                y = df["SalesUnides"]
                mean_y = y.mean()

                if mode == "auto":
                    if "Engagement" in df.columns and df["Engagement"].sum() == 0:
                        mode = "llp"
                    else:
                        mode = "bluey"

                seasonalities = [col for col in ["ChildrensDay", "xMas", "BuenFin", "ValentinesDay"] if col in df.columns]

                if mode == "llp":
                    w_search = (0.30 * mean_y) / df["Searching"].mean() if "Searching" in df.columns and df["Searching"].mean() != 0 else 0
                    w_precio = (0.28 * mean_y) / df["PrecioPromedio"].mean() if "PrecioPromedio" in df.columns and df["PrecioPromedio"].mean() != 0 else 0
                    w_menciones = (0.13 * mean_y) / df["Menciones"].mean() if "Menciones" in df.columns and df["Menciones"].mean() != 0 else 0
                    w_engagement = 0.0
                    w_resenias = (0.10 * mean_y) / df["Resenias"].mean() if "Resenias" in df.columns and df["Resenias"].mean() != 0 else 0
                    target_base_pct = 0.19
                else:
                    w_search = (0.13 * mean_y) / df["Searching"].mean() if "Searching" in df.columns and df["Searching"].mean() != 0 else 0
                    w_precio = (0.11 * mean_y) / df["PrecioPromedio"].mean() if "PrecioPromedio" in df.columns and df["PrecioPromedio"].mean() != 0 else 0
                    w_resenias = (0.15 * mean_y) / df["Resenias"].mean() if "Resenias" in df.columns and df["Resenias"].mean() != 0 else 0

                    if "Menciones" in df.columns and "Engagement" in df.columns and df["Engagement"].sum() > 0:
                        lr_me = LinearRegression(fit_intercept=False)
                        lr_me.fit(df[["Menciones", "Engagement"]], y)
                        tot_c_me = (lr_me.coef_[0] * df["Menciones"].mean()) + (lr_me.coef_[1] * df["Engagement"].mean())
                        scale_me = (0.32 * mean_y) / tot_c_me if tot_c_me != 0 else 1
                        w_menciones = lr_me.coef_[0] * scale_me
                        w_engagement = lr_me.coef_[1] * scale_me
                    else:
                        w_menciones = (0.32 * mean_y) / df["Menciones"].mean() if "Menciones" in df.columns and df["Menciones"].mean() != 0 else 0
                        w_engagement = 0.0

                    target_base_pct = 0.29

                df["Intercept"] = 1
                X_base_cols = ["Intercept"] + seasonalities
                lr_base = LinearRegression(fit_intercept=False)
                lr_base.fit(df[X_base_cols], y)

                tot_c_base = (lr_base.coef_[0] * 1)
                for i, col in enumerate(seasonalities):
                    tot_c_base += lr_base.coef_[i+1] * df[col].mean()

                scale_base = (target_base_pct * mean_y) / tot_c_base if tot_c_base != 0 else 1
                w_intercept = lr_base.coef_[0] * scale_base
                w_season = {col: lr_base.coef_[i+1] * scale_base for i, col in enumerate(seasonalities)}

                growth_index = (
                    (df["Searching"] * w_search if "Searching" in df else 0) +
                    (df["PrecioPromedio"] * w_precio if "PrecioPromedio" in df else 0) +
                    (df["Menciones"] * w_menciones if "Menciones" in df else 0) +
                    (df["Engagement"] * w_engagement if "Engagement" in df else 0) +
                    (df["Resenias"] * w_resenias if "Resenias" in df else 0) +
                    w_intercept
                )
                for col in seasonalities:
                    growth_index += df[col] * w_season[col]

                df["Growth_Index"] = growth_index

                corr_val, _ = pearsonr(growth_index, y) if len(growth_index) > 1 else (0, 0)
                r_sq = r2_score(y, growth_index) if len(growth_index) > 1 else 0

                grouped_contribs = {
                    "Baseline (Mid/Long Term)": (w_intercept * 1) + sum([w_season[col] * df[col].mean() for col in seasonalities if col in w_season]),
                    "Conversiones & SoL": (w_menciones * (df["Menciones"].mean() if "Menciones" in df else 0)) + (w_engagement * (df["Engagement"].mean() if "Engagement" in df else 0)),
                    "Búsquedas y SoS": (w_search * (df["Searching"].mean() if "Searching" in df else 0)) + (w_resenias * (df["Resenias"].mean() if "Resenias" in df else 0)),
                    "Precio y Promociones": w_precio * (df["PrecioPromedio"].mean() if "PrecioPromedio" in df else 0)
                }

                weights_dict = {
                    "w_search": w_search,
                    "w_precio": w_precio,
                    "w_menciones": w_menciones,
                    "w_engagement": w_engagement,
                    "w_resenias": w_resenias,
                    "w_intercept": w_intercept,
                    "w_season": w_season
                }

                return df, weights_dict, corr_val, r_sq, grouped_contribs

            df_model, weights, correlation_r, r2_val, grouped_contribs = fit_growth_index_model(raw_df, model_mode)

            col_inputs, col_results = st.columns([1.1, 2.5])

            with col_inputs:
                st.subheader("Variables Observables (Inputs)")

                with st.container(border=True):
                    st.markdown("💬 **1. Conversaciones & SoL**")
                    menciones_val = st.slider("Menciones Absolutas", 0, 10000, int(df_model["Menciones"].mean()) if "Menciones" in df_model else 1000, 50, key="slider_menciones")
                    engagement_val = st.slider("Engagement Rate / Absoluto", 0.0, 0.0030, float(df_model["Engagement"].mean()) if "Engagement" in df_model else 0.0005, 0.0001, format="%.6f", key="slider_engagement")

                with st.container(border=True):
                    st.markdown("🔍 **2. Búsquedas & SoS**")
                    searching_val = st.slider("Share of Search (Índice)", 0, 200, int(df_model["Searching"].mean()) if "Searching" in df_model else 60, 1, key="slider_searching")
                    resenias_val = st.slider("Reseñas / Validación", 0, 200, int(df_model["Resenias"].mean() if "Resenias" in df_model else 50), 1, key="slider_resenias")

                with st.container(border=True):
                    st.markdown("💰 **3. Precio y Promociones**")
                    precio_val = st.slider("Precio Promedio ($ MXN)", 100, 1000, int(df_model["PrecioPromedio"].mean()) if "PrecioPromedio" in df_model else 400, 5, key="slider_precio")

                st.markdown("**📅 Estacionalidad / Temporada Activa**")
                active_child = st.checkbox("Día del Niño", value=False, key="chk_child")
                active_xmas = st.checkbox("Navidad / Q4", value=False, key="chk_xmas")
                active_buenfin = st.checkbox("Buen Fin", value=False, key="chk_buenfin")
                active_val = st.checkbox("San Valentín", value=False, key="chk_val")

            sim_gi = (
                searching_val * weights["w_search"] +
                precio_val * weights["w_precio"] +
                menciones_val * weights["w_menciones"] +
                engagement_val * weights["w_engagement"] +
                resenias_val * weights["w_resenias"] +
                weights["w_intercept"] +
                (weights["w_season"].get("ChildrensDay", 0) if active_child else 0) +
                (weights["w_season"].get("xMas", 0) if active_xmas else 0) +
                (weights["w_season"].get("BuenFin", 0) if active_buenfin else 0) +
                (weights["w_season"].get("ValentinesDay", 0) if active_val else 0)
            )

            baseline_portion = weights["w_intercept"] + (
                (weights["w_season"].get("ChildrensDay", 0) if active_child else 0) +
                (weights["w_season"].get("xMas", 0) if active_xmas else 0) +
                (weights["w_season"].get("BuenFin", 0) if active_buenfin else 0) +
                (weights["w_season"].get("ValentinesDay", 0) if active_val else 0)
            )
            incremental_portion = sim_gi - baseline_portion

            with col_results:
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Variable Latente (GI)", f"{sim_gi:.1f} pts")
                kpi2.metric("Baseline (Mid/Long)", f"{baseline_portion:,.0f} uds")
                kpi3.metric("Incremental (GI Drivers)", f"+{incremental_portion:,.0f} uds")
                kpi4.metric("Correlación (Pearson R)", f"R = {correlation_r:.4f}", f"R² = {r2_val:.4f}")

                st.markdown("---")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    tot_sum_contrib = sum(grouped_contribs.values()) if sum(grouped_contribs.values()) != 0 else 1
                    contrib_pcts = {k: (v / tot_sum_contrib) * 100 for k, v in grouped_contribs.items()}
                    df_contrib_plot = pd.DataFrame(list(contrib_pcts.items()), columns=["Grupo", "Porcentaje"]).sort_values("Porcentaje", ascending=True)

                    fig_contrib = go.Figure(go.Bar(
                        x=df_contrib_plot["Porcentaje"],
                        y=df_contrib_plot["Grupo"],
                        orientation='h',
                        marker=dict(color=['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b']),
                        text=[f"{val:.1f}%" for val in df_contrib_plot["Porcentaje"]],
                        textposition='auto'
                    ))
                    fig_contrib.update_layout(
                        title="<b>Contribución Econométrica por Grupo (%)</b>",
                        xaxis_title="Porcentaje de Contribución (%)",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_contrib, use_container_width=True)

                with chart_col2:
                    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
                    x_weeks = [f"Sem {i+1}" for i in range(len(df_model))]

                    fig_trend.add_trace(go.Scatter(
                        x=x_weeks, y=df_model["SalesUnides"], name="Ventas Reales (Unidades)",
                        line=dict(color='#10b981', width=3)
                    ), secondary_y=False)

                    fig_trend.add_trace(go.Scatter(
                        x=x_weeks, y=df_model["Growth_Index"], name="Growth Index (GI)",
                        line=dict(color='#4f46e5', width=3, dash='dash')
                    ), secondary_y=True)

                    fig_trend.add_trace(go.Scatter(
                        x=["Simulación Actual"], y=[sim_gi], name="Simulación GI",
                        mode='markers', marker=dict(size=14, color='#f59e0b', symbol='star')
                    ), secondary_y=True)

                    fig_trend.update_layout(
                        title="<b>Ajuste de Serie Temporal: GI vs Ventas</b>",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=df_model["Growth_Index"], y=df_model["SalesUnides"],
                    mode='markers', name='Semanas Históricas',
                    marker=dict(color='#4f46e5', size=9, opacity=0.8)
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=[sim_gi], y=[sim_gi],
                    mode='markers', name='Simulación Actual',
                    marker=dict(color='#f59e0b', symbol='star', size=16)
                ))
                fig_scatter.update_layout(
                    title=f"<b>Matriz de Dispersión y Predictibilidad (R² = {r2_val:.4f} | Pearson R = {correlation_r:.4f})</b>",
                    xaxis_title="Growth Index Predictivo",
                    yaxis_title="Ventas (Unidades)",
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")

with tab2:
    st.markdown("### 📈 MiA - Medición del Impacto de Medios")
    st.markdown("Sube tu archivo Excel con los resultados del MMM (pestañas: `Contribution`, `Spend`, `Revenue`, `Curvas S`) para visualizar las contribuciones, CP-GI, ROI, Curvas S y la evolución en áreas apiladas.")

    mmm_uploaded_file = st.file_uploader("Subir Excel de Resultados MMM (ej. Resultados_MiA_Bluey.xlsx)", type=["xlsx", "xls"], key="mmm_excel_uploader")
    
    if mmm_uploaded_file is not None:
        try:
            xls = pd.ExcelFile(mmm_uploaded_file)
            sheet_names = xls.sheet_names
            
            df_contrib_mmm = pd.read_excel(mmm_uploaded_file, sheet_name="Contribution") if "Contribution" in sheet_names else None
            df_spend_mmm = pd.read_excel(mmm_uploaded_file, sheet_name="Spend") if "Spend" in sheet_names else None
            df_revenue_mmm = pd.read_excel(mmm_uploaded_file, sheet_name="Revenue") if "Revenue" in sheet_names else None
            df_curvas_mmm = pd.read_excel(mmm_uploaded_file, sheet_name="Curvas S") if "Curvas S" in sheet_names else None
            
            if df_curvas_mmm is not None and not df_curvas_mmm.empty:
                df_curvas_mmm.columns = [c.strip() for c in df_curvas_mmm.columns]
                st.session_state['df_curvas_mmm'] = df_curvas_mmm

            st.success(f"Archivo Excel **{mmm_uploaded_file.name}** cargado correctamente.")

            if df_contrib_mmm is not None and not df_contrib_mmm.empty:
                area_cols = [c for c in df_contrib_mmm.columns if c not in ["DateWeekClose", "Total Media"]]

                st.markdown("---")
                st.subheader("🌊 Contribución de los Medios y Baseline en Forma Tendencial (Áreas Apiladas)")
                fig_area = go.Figure()
                x_date = df_contrib_mmm["DateWeekClose"] if "DateWeekClose" in df_contrib_mmm.columns else df_contrib_mmm.index

                color_palette = ["#94a3b8", "#0668E1", "#F59E0B", "#10B981", "#8B5CF6", "#FF0000"]
                for idx, col in enumerate(area_cols):
                    c_color = color_palette[idx % len(color_palette)]
                    fig_area.add_trace(go.Scatter(
                        x=x_date, y=df_contrib_mmm[col], name=col,
                        mode="lines", stackgroup="one", line=dict(width=0.5, color=c_color)
                    ))
                fig_area.update_layout(
                    title="<b>Evolución Temporal de Contribución (Baseline + Medios) al GI</b>",
                    xaxis_title="Fecha / Semana",
                    yaxis_title="Contribución Acumulada",
                    height=380,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_area, use_container_width=True)

                bar_media_cols = [c for c in df_contrib_mmm.columns if c not in ["DateWeekClose", "Baseline", "Total Media"]]

                sum_contrib = df_contrib_mmm[bar_media_cols].sum()
                if "Total Media" in df_contrib_mmm.columns:
                    sum_contrib["Total Media"] = df_contrib_mmm["Total Media"].sum()

                sum_spend = df_spend_mmm[bar_media_cols].sum() if df_spend_mmm is not None and not df_spend_mmm.empty else pd.Series(0, index=bar_media_cols)
                if df_spend_mmm is not None and "Total Media" in df_spend_mmm.columns:
                    sum_spend["Total Media"] = df_spend_mmm["Total Media"].sum()

                sum_revenue = df_revenue_mmm[bar_media_cols].sum() if df_revenue_mmm is not None and not df_revenue_mmm.empty else pd.Series(0, index=bar_media_cols)
                if df_revenue_mmm is not None and "Total Media" in df_revenue_mmm.columns:
                    sum_revenue["Total Media"] = df_revenue_mmm["Total Media"].sum()

                st.markdown("---")
                st.subheader("📊 Desempeño General a Total Periodo (Contribution, CP-GI & ROI)")

                bcol1, bcol2, bcol3 = st.columns(3)

                df_bar_contrib = sum_contrib.reset_index()
                df_bar_contrib.columns = ["Medio", "Contribucion"]
                fig_bar_c = px.bar(df_bar_contrib, x="Medio", y="Contribucion", title="Contribución Total Periodo", color="Medio", template="plotly_white")
                fig_bar_c.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
                bcol1.plotly_chart(fig_bar_c, use_container_width=True)

                cp_gi = sum_spend / sum_contrib if sum_contrib.sum() != 0 else pd.Series(0, index=sum_contrib.index)
                df_bar_cp = cp_gi.reset_index()
                df_bar_cp.columns = ["Medio", "CPGI"]
                fig_bar_cp = px.bar(df_bar_cp, x="Medio", y="CPGI", title="Costo por Punto (CP-GI)", color="Medio", template="plotly_white")
                fig_bar_cp.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
                bcol2.plotly_chart(fig_bar_cp, use_container_width=True)

                roi_val = sum_revenue / sum_spend if sum_spend.sum() != 0 else pd.Series(0, index=sum_spend.index)
                df_bar_roi = roi_val.reset_index()
                df_bar_roi.columns = ["Medio", "ROI"]
                fig_bar_roi = px.bar(df_bar_roi, x="Medio", y="ROI", title="Retorno de Inversión (ROI)", color="Medio", template="plotly_white")
                fig_bar_roi.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
                bcol3.plotly_chart(fig_bar_roi, use_container_width=True)

                if df_curvas_mmm is not None and not df_curvas_mmm.empty:
                    st.markdown("---")
                    st.subheader("📈 Curvas de Respuesta (S-Curves) y mROI (Empíricas)")
                    scol1, scol2 = st.columns(2)

                    fig_s = px.line(df_curvas_mmm, x="Inversión", y="Contribución", color="Media", title="Curvas S de Respuesta por Medio", template="plotly_white")
                    fig_s.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    scol1.plotly_chart(fig_s, use_container_width=True)

                    fig_mroi = px.line(df_curvas_mmm, x="Inversión", y="mROI", color="Media", title="Campanas de mROI Marginal", template="plotly_white")
                    fig_mroi.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    scol2.plotly_chart(fig_mroi, use_container_width=True)

            else:
                st.warning("El archivo Excel cargado no contiene la pestaña 'Contribution'.")

        except Exception as e:
            st.error(f"Error procesando el archivo Excel de MMM: {e}")
    else:
        st.info("👆 Por favor, carga el archivo Excel de resultados MMM (ej. `Resultados_MiA_Bluey.xlsx`) para visualizar las gráficas de área, contribución, CP-GI y ROI.")

with tab3:
    st.markdown("### ⚡ Budget Allocator")
    
    sub_tab1, sub_tab2 = st.tabs(["🎛️ 1. Simulador Manual & Plan de Medios", "🤖 2. Optimizador Avanzado (Media Plan & Benchmark)"])

    interpolators = {}
    df_curvas_loaded = st.session_state.get('df_curvas_mmm', None)
    
    for key, param in MEDIA_PARAMS.items():
        media_name = param['name']
        if df_curvas_loaded is not None and not df_curvas_loaded.empty:
            df_m = df_curvas_loaded[df_curvas_loaded['Media'].str.strip().str.lower() == media_name.lower()].sort_values('Inversión')
        else:
            df_m = pd.DataFrame()

        if not df_m.empty:
            x = df_m['Inversión'].values
            y_contrib = df_m['Contribución'].values
            y_mroi = df_m['mROI'].values
        else:
            x = np.linspace(0, 2000000, 100)
            y_contrib = x * 0.00001
            y_mroi = np.ones_like(x) * 0.5

        interpolators[key] = {
            'contrib': interp1d(x, y_contrib, kind='linear', bounds_error=False, fill_value=(0, y_contrib[-1])),
            'mroi': interp1d(x, y_mroi, kind='linear', bounds_error=False, fill_value=(y_mroi[-1], y_mroi[0]))
        }

    with sub_tab1:
        st.markdown("Configura el presupuesto y las semanas en aire para cada canal. Visualiza la asignación de inversión y su distribución semanal apilada.")
        opt_col, res_col = st.columns([1, 2.5])

        with opt_col:
            st.subheader("Configuración de Medios")
            for key, param in MEDIA_PARAMS.items():
                with st.container(border=True):
                    st.markdown(f"**<span style='color:{param['color']}'>●</span> {param['name']}**", unsafe_allow_html=True)
                    st.session_state.opt_spends[key] = st.slider("Spend ($)", 0, 3000000, st.session_state.opt_spends[key], 10000, key=f"os_{key}", label_visibility="collapsed")
                    st.session_state.opt_weeks[key] = st.slider("Semanas", 1, 52, st.session_state.opt_weeks[key], key=f"ow_{key}")

        optimizerSpend = st.session_state.opt_spends.copy()
        weeksOnAir = st.session_state.opt_weeks.copy()
        opt_total = sum(optimizerSpend.values())

        total_contrib = sum([float(interpolators[k]['contrib'](optimizerSpend[k])) for k in MEDIA_PARAMS])
        total_revenue = total_contrib * REVENUE_PER_GI
        current_roi = (total_revenue / opt_total) if opt_total > 0 else 0

        with res_col:
            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Inversión Total", f"${opt_total:,.0f}")
            rcol2.metric("ROI Proyectado", f"{current_roi:.2f}x")

            c1, c2 = st.columns(2)
            
            mixData = []
            for key, param in MEDIA_PARAMS.items():
                mixData.append({
                    'Media': param['name'],
                    'Inversión': optimizerSpend[key],
                    'Color': param['color']
                })
            df_mix = pd.DataFrame(mixData)

            fig_bar = px.bar(
                df_mix, x='Media', y='Inversión', color='Media',
                color_discrete_map={param['name']: param['color'] for param in MEDIA_PARAMS.values()},
                title="Presupuesto Asignado por Medio",
                template="plotly_white"
            )
            fig_bar.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            c1.plotly_chart(fig_bar, use_container_width=True)

            maxWeeks = max(weeksOnAir.values()) if weeksOnAir.values() else 1
            weeks_list = [f"Sem {w+1}" for w in range(maxWeeks)]
            df_weekly_spend = pd.DataFrame({'Semana': weeks_list})

            for key, param in MEDIA_PARAMS.items():
                cSpend = optimizerSpend[key]
                cWeeks = weeksOnAir[key]
                weekly_spends = [0] * maxWeeks
                if cWeeks > 0 and cSpend > 0:
                    peakWeek = max(1, round(cWeeks * 0.35))
                    sumWeights = sum([math.exp(-((w - peakWeek)**2) / (cWeeks * 0.5)) for w in range(1, cWeeks + 1)])
                    for w in range(1, cWeeks + 1):
                        if w <= cWeeks:
                            weight = math.exp(-((w - peakWeek)**2) / (cWeeks * 0.5)) / sumWeights
                            weekly_spends[w-1] = cSpend * weight
                df_weekly_spend[param['name']] = weekly_spends

            fig_area = go.Figure()
            for key, param in MEDIA_PARAMS.items():
                fig_area.add_trace(go.Scatter(
                    x=df_weekly_spend['Semana'], y=df_weekly_spend[param['name']],
                    name=param['name'], mode='lines', stackgroup='one',
                    line=dict(width=0.5, color=param['color'])
                ))
            fig_area.update_layout(
                title="Evolución de Inversión Semanal (Áreas Apiladas)",
                xaxis_title="Semanas",
                yaxis_title="Inversión ($)",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            c2.plotly_chart(fig_area, use_container_width=True)

    with sub_tab2:
        st.markdown("Sube tu **Plan de Medios en Excel** (`MediaPlan_LLP.xlsx`) y tu **Benchmark de Inversión** (`Benchmark_Budget_LLP.xlsx`), e ingresa el presupuesto total de la campaña para optimizar la distribución manteniendo continuidad, optimalidad y heavy up.")

        col_up1, col_up2, col_up3 = st.columns(3)
        mp_file = col_up1.file_uploader("📂 Subir Plan de Medios (Excel)", type=["xlsx", "xls"], key="media_plan_excel")
        bench_file = col_up2.file_uploader("📂 Subir Benchmark de Inversión (Excel)", type=["xlsx", "xls"], key="benchmark_excel")
        campaign_budget_input = col_up3.number_input("💰 Presupuesto Total Campaña ($)", min_value=100000, max_value=100000000, value=15000000, step=100000, key="camp_budget_num")

        if mp_file is not None and bench_file is not None:
            try:
                df_mp_orig = pd.read_excel(mp_file)
                df_bench_orig = pd.read_excel(bench_file)

                st.success("¡Plan de Medios y Benchmark cargados correctamente!")

                sustain_b = float(df_bench_orig.loc[0, 'Sustain'])
                optimal_b = float(df_bench_orig.loc[0, 'Optimal'])
                heavy_up_b = float(df_bench_orig.loc[0, 'Heavy Up'])

                st.info(f"📌 **Benchmarks detectados** -> Sustain (Mínimo): ${sustain_b:,.0f} | Optimal: ${optimal_b:,.0f} | Heavy Up: ${heavy_up_b:,.0f}")

                chan_col = df_mp_orig.columns[0]
                date_cols = [c for c in df_mp_orig.columns if c != chan_col]

                original_spends_per_channel = {}
                for idx, row in df_mp_orig.iterrows():
                    c_name = str(row[chan_col]).strip()
                    total_c_spend = row[date_cols].sum()
                    original_spends_per_channel[c_name] = total_c_spend

                orig_total_spend = sum(original_spends_per_channel.values())
                orig_contrib = sum([float(interpolators[k]['contrib'](original_spends_per_channel.get(param['name'], 0))) for k, param in MEDIA_PARAMS.items()])
                orig_revenue = orig_contrib * REVENUE_PER_GI
                orig_roi = (orig_revenue / orig_total_spend) if orig_total_spend > 0 else 0

                curr_opt_spends = {k: campaign_budget_input / len(MEDIA_PARAMS) for k in MEDIA_PARAMS}
                for _ in range(50):
                    marginalRois = []
                    for k in MEDIA_PARAMS:
                        sp = curr_opt_spends[k]
                        mR = float(interpolators[k]['mroi'](sp))
                        marginalRois.append({'key': k, 'mRoi': mR})
                    marginalRois.sort(key=lambda x: x['mRoi'], reverse=True)
                    best_k = marginalRois[0]['key']
                    worst_k = marginalRois[-1]['key']
                    step = campaign_budget_input * 0.01
                    if curr_opt_spends[worst_k] > sustain_b * len(date_cols):
                        curr_opt_spends[worst_k] -= step
                        curr_opt_spends[best_k] += step

                opt_total_spend_final = sum(curr_opt_spends.values())
                opt_contrib = sum([float(interpolators[k]['contrib'](curr_opt_spends[k])) for k in MEDIA_PARAMS])
                opt_revenue = opt_contrib * REVENUE_PER_GI
                opt_roi = (opt_revenue / opt_total_spend_final) if opt_total_spend_final > 0 else 0

                st.markdown("---")
                st.subheader("📊 Resultados Resumidos: Contribution Total de Media & ROI")

                res_k1, res_k2, res_k3, res_k4 = st.columns(4)
                res_k1.metric("Inversión Total Campaña", f"${campaign_budget_input:,.0f}")
                res_k2.metric("Contribution Total (GI)", f"{opt_contrib:,.1f} pts", f"Orig: {orig_contrib:.1f}")
                res_k3.metric("ROI Proyectado (Opt)", f"{opt_roi:.2f}x", f"Orig: {orig_roi:.2f}x")
                res_k4.metric("Lift en Revenue", f"+${(opt_revenue - orig_revenue)/1000000:,.2f}M")

                comp_col1, comp_col2 = st.columns(2)

                df_comp_mix = pd.DataFrame([
                    {'Media': param['name'], 'Plan Original': original_spends_per_channel.get(param['name'], 0), 'Plan Optimizado': curr_opt_spends[key]}
                    for key, param in MEDIA_PARAMS.items()
                ])

                fig_comp = px.bar(
                    df_comp_mix, x='Media', y=['Plan Original', 'Plan Optimizado'],
                    barmode='group', title="Comparativa de Inversión: Plan Original vs. Plan Optimizado",
                    template="plotly_white"
                )
                fig_comp.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
                comp_col1.plotly_chart(fig_comp, use_container_width=True)

                df_opt_weekly = pd.DataFrame({'Semana': [str(d)[:10] for d in date_cols]})
                for key, param in MEDIA_PARAMS.items():
                    ch_name = param['name']
                    orig_channel_weekly = df_mp_orig[df_mp_orig[chan_col].str.strip().str.lower() == ch_name.lower()][date_cols].values.flatten()
                    sum_orig_w = sum(orig_channel_weekly) if sum(orig_channel_weekly) > 0 else 1
                    opt_weekly_channels = [curr_opt_spends[key] * (w / sum_orig_w) for w in orig_channel_weekly]
                    df_opt_weekly[ch_name] = opt_weekly_channels

                fig_opt_area = go.Figure()
                for key, param in MEDIA_PARAMS.items():
                    fig_opt_area.add_trace(go.Scatter(
                        x=df_opt_weekly['Semana'], y=df_opt_weekly[param['name']],
                        name=param['name'], mode='lines', stackgroup='one',
                        line=dict(width=0.5, color=param['color'])
                    ))
                fig_opt_area.update_layout(
                    title="Evolución Semanal del Plan Optimizado (Áreas Apiladas)",
                    xaxis_title="Fecha de Semana",
                    yaxis_title="Inversión ($)",
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                comp_col2.plotly_chart(fig_opt_area, use_container_width=True)

            except Exception as e:
                st.error(f"Error procesando los archivos del plan de medios o benchmark: {e}")
        else:
            st.info("👆 Por favor, sube ambos archivos Excel (`MediaPlan_LLP.xlsx` y `Benchmark_Budget_LLP.xlsx`) e ingresa el presupuesto de la campaña para ejecutar el optimizador avanzado.")