# ==============================================================================
# PROYECTO INTEGRADOR I: PANEL INTERACTIVO CON STREAMLIT
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. CONFIGURACIÓN Y TÍTULO ---
# Configura el layout para usar el ancho completo de la página
st.set_page_config(layout="wide", page_title="EV Data Analysis")

st.title("🚗 PROYECTO INTEGRADOR I: Análisis Avanzado de Datos de Autos Eléctricos")
st.markdown("---")
st.header("Panel Interactivo Basado en Datos Limpios y Preparados")
st.caption("Los datos han sido sometidos a Imputación MICE, Capping IQR y Reducción de Dimensionalidad (PCA).")

# --- 2. CARGA DEL DATA SET LIMPIO ---
FILE_PATH_PREPARED = "C:\\Users\\juanf\\OneDrive\\Documentos\\Proyecto_integrador-trabajo-\\df_final_preparado.csv"

@st.cache_data
def load_data(path):
    """Carga y prepara el DataFrame limpio."""
    try:
        df = pd.read_csv(path)
        # Asegurarse de que 'year' sea entero para el slider
        df['year'] = df['year'].astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos limpio '{path}'.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.stop()

df = load_data(FILE_PATH_PREPARED)


# --- 3. BARRA LATERAL (SIDEBAR) Y FILTROS ---
st.sidebar.title("🛠️ Controles de Filtro")

# SLIDER DE AÑO
min_year = int(df['year'].min())
max_year = int(df['year'].max())
year_selection = st.sidebar.slider(
    'Seleccionar Rango de Años',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# FILTRO DE REGIÓN
region_options = sorted(df['region'].unique())
selected_regions = st.sidebar.multiselect(
    'Seleccionar Región(es)',
    options=region_options,
    default=region_options # Por defecto, todas seleccionadas
)

# APLICAR FILTROS
df_filtered = df[
    (df['year'] >= year_selection[0]) &
    (df['year'] <= year_selection[1]) &
    (df['region'].isin(selected_regions))
]

# Mensaje de advertencia si el filtro es vacío
if df_filtered.empty:
    st.warning("No hay datos que coincidan con los filtros seleccionados. Por favor, ajusta los parámetros.")
    st.stop()


# --- 4. MÉTRICAS CLAVE (KPIS) ---
st.subheader("📊 Métricas Clave (KPIs)")

col1, col2, col3, col4 = st.columns(4)

total_rows = len(df_filtered)
# Usamos median o mean dependiendo de la robustez, pero mean es común para KPIs
avg_price = df_filtered['price'].mean()
avg_range = df_filtered['range_km'].mean()
# 'value' es la métrica principal (stock, ventas, etc.)
total_value = df_filtered['value'].sum()

col1.metric("Registros Filtrados", f"{total_rows:,.0f}")
col2.metric("Precio Promedio Ajustado (USD)", f"${avg_price:,.0f}")
col3.metric("Autonomía Promedio (km)", f"{avg_range:,.0f} km")
col4.metric("Valor Total Agregado", f"{total_value:,.0f}", delta="Historical Agg.")

st.markdown("---")

# --- 5. VISUALIZACIONES AVANZADAS (PCA y Tendencias) ---

# 5.1 ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
st.header("1. Análisis de Componentes Principales (PCA)")
st.caption("Visualización de las especificaciones del vehículo reducidas a dos componentes principales.")

# Selector para colorear los puntos
pca_color_by = st.selectbox(
    'Colorear Puntos por:',
    options=['region', 'mode', 'powertrain'],
    index=0
)

# Filtro adicional de 'mode' para que el gráfico no esté demasiado cargado
pca_mode_options = sorted(df_filtered['mode'].unique())
pca_selected_mode = st.selectbox(
    'Filtrar MODO (Cars, Vans, etc.) para el Gráfico PCA',
    options=pca_mode_options,
    index=pca_mode_options.index('Cars') if 'Cars' in pca_mode_options else 0
)

df_pca_plot = df_filtered[df_filtered['mode'] == pca_selected_mode]

fig_pca = px.scatter(
    df_pca_plot,
    x='PC1_Car_Specs',
    y='PC2_Car_Specs',
    color=pca_color_by,
    hover_data=['year', 'powertrain', 'price', 'range_km', 'battery_capacity'],
    title=f"PCA: PC1 vs PC2 para el modo '{pca_selected_mode}'",
    labels={'PC1_Car_Specs': 'Componente Principal 1', 'PC2_Car_Specs': 'Componente Principal 2'},
    height=550
)
st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("---")

# 5.2 TENDENCIA HISTÓRICA
st.header("2. Tendencia Histórica de la Métrica 'Value'")
st.caption("Muestra la evolución del Stock, Ventas, o el parámetro 'Value' a lo largo del tiempo, agrupado por tipo de tren motriz.")

# Agrupar por año y powertrain
trend_data = df_filtered.groupby(['year', 'powertrain']).agg(
    total_value=('value', 'sum')
).reset_index()

fig_trend = px.line(
    trend_data,
    x='year',
    y='total_value',
    color='powertrain',
    markers=True,
    title='Agregado de "Value" a lo largo del tiempo por Powertrain',
    height=500
)
fig_trend.update_layout(xaxis_title="Año", yaxis_title="Valor Agregado (Stock/Ventas/etc.)")
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# 5.3 DISTRIBUCIÓN DE VARIABLES CLAVE (POST-LIMPIEZA)
st.header("3. Distribución de Variables Numéricas Clave")
st.caption("Visualiza el impacto de la limpieza (imputación y capping de outliers) en las variables.")

distribution_col = st.selectbox(
    'Seleccionar Variable de Distribución',
    options=['price', 'range_km', 'battery_capacity', 'weight_kg', 'motor_power']
)

fig_dist = px.histogram(
    df_filtered,
    x=distribution_col,
    nbins=30,
    title=f'Distribución de {distribution_col} (Ajustado)',
    color_discrete_sequence=['darkblue'],
    height=450
)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# --- 6. DATA SET FINAL LIMPIO ---
st.header("Data Set Final Limpio y Preparado (Vista Previa)")
st.caption(f"Mostrando los primeros {len(df_filtered)} registros filtrados.")
st.dataframe(df_filtered)