# =================================================================================
# PROYECTO INTEGRADOR I: LIMPIEZA Y PREPARACIÓN AVANZADA DE DATOS DE AUTOS ELÉCTRICOS
# =================================================================================

# --- 1. CONFIGURACIÓN E IMPORTACIONES ---
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import iqr

# Nombres de archivos
FILE_PATH = "C:\\Users\\juanf\\OneDrive\\Documentos\\Proyecto_integrador-trabajo-\\IEA Global EV Data 2024 full.csv"
OUTPUT_FILE = "df_final_preparado.csv"

# Columnas clave para la limpieza avanzada
IMPUTATION_COLS = [
    'price', 'range_km', 'charging_time', 'sales_volume', 'co2_saved',
    'battery_capacity', 'energy_efficiency', 'weight_kg',
    'number_of_seats', 'motor_power', 'distance_traveled'
]
CATEGORICAL_COLS = ['region', 'parameter', 'mode', 'powertrain']

print("--- INICIO DEL PROCESO DE LIMPIEZA AVANZADA ---")

# --- 2. PASO 1 & 2: LIMPIEZA INICIAL, FILTRADO Y CODIFICACIÓN ---

# 2.1. Carga de datos y Manejo de Duplicados
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{FILE_PATH}'.")
    exit()

initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"Filas iniciales: {initial_rows} -> Filas después de eliminar duplicados: {len(df)}")

# 2.2. Filtrado Histórico (Esencial para la precisión de los modelos)
df_historical = df[df['category'] == 'Historical'].copy()
df_historical.drop(columns=['unit', 'category'], inplace=True)
print(f"Filas después de filtrar 'Historical': {len(df_historical)}")

# Guardamos las variables categóricas originales antes de la codificación
df_context = df_historical[CATEGORICAL_COLS + ['year', 'value']].reset_index(drop=True)

# 2.3. Codificación One-Hot Encoding (Necesario para MICE y PCA)
df_encoded = pd.get_dummies(df_historical, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
print(f"DataFrame codificado creado. Columnas totales: {df_encoded.shape[1]}")


# --- 3. PASO 3: IMPUTACIÓN AVANZADA (MAECI/MICE) ---

# Identificamos todas las columnas numéricas para la imputación
cols_to_impute = df_encoded.select_dtypes(include=np.number).columns.tolist()

# Inicializar IterativeImputer (implementación de MICE/MAECI)
# Utiliza un modelo de regresión (por defecto, BayesianRidge) para estimar los nulos.
mice_imputer = IterativeImputer(random_state=42, max_iter=10)

# Aplicar la imputación MICE al subconjunto de columnas numéricas
df_imputed_array = mice_imputer.fit_transform(df_encoded[cols_to_impute])

# Reconstruir el DataFrame imputado
df_imputed = pd.DataFrame(df_imputed_array, columns=cols_to_impute)
print("\n--- Imputación MAECI/MICE Completada ---")


# --- 4. PASO 4: MANEJO DE OUTLIERS (IQR CAPPING) ---

# Columnas numéricas clave para la detección y ajuste de outliers
outlier_check_cols = ['price', 'range_km', 'battery_capacity', 'weight_kg', 'motor_power']

for col in outlier_check_cols:
    # 4.1. Detección: Calcular Q1, Q3 y IQR
    Q1 = df_imputed[col].quantile(0.25)
    Q3 = df_imputed[col].quantile(0.75)
    IQR_val = Q3 - Q1

    # 4.2. Manejo: Definir límites para el "Capping" (ajuste)
    lower_limit = Q1 - 1.5 * IQR_val
    upper_limit = Q3 + 1.5 * IQR_val

    # Aplicar Capping: Reemplazar valores fuera de los límites por los límites
    df_imputed[col] = np.where(df_imputed[col] < lower_limit, lower_limit, df_imputed[col])
    df_imputed[col] = np.where(df_imputed[col] > upper_limit, upper_limit, df_imputed[col])

print("\n--- Manejo de Outliers (Capping con IQR) Completado ---")


# --- 5. PASO 4.2: ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) ---

# Variables a estandarizar y usar en PCA
pca_features = ['price', 'range_km', 'battery_capacity', 'weight_kg', 'motor_power']

# 5.1. Estandarización de los datos (Escalamiento)
scaler = StandardScaler()
df_pca_scaled = scaler.fit_transform(df_imputed[pca_features])

# 5.2. Aplicación de PCA
# Reducimos a 2 componentes para la visualización en Streamlit.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_pca_scaled)

# 5.3. Integración de Componentes al DataFrame
df_imputed['PC1_Car_Specs'] = principal_components[:, 0]
df_imputed['PC2_Car_Specs'] = principal_components[:, 1]

print(f"\n--- PCA Completado ---")
print(f"Varianza total explicada (PC1+PC2): {pca.explained_variance_ratio_.sum():.2f}")


# --- 6. PREPARACIÓN DEL DATASET FINAL ---

# Reconstruimos el DataFrame final uniendo las nuevas columnas PCA con las columnas de contexto.
df_final = df_context.copy()

# Copiamos las variables imputadas, ajustadas (outliers) y usadas en PCA
for col in pca_features + IMPUTATION_COLS:
    if col in df_imputed.columns:
        df_final[col] = df_imputed[col]
        
# Añadimos los componentes principales
df_final['PC1_Car_Specs'] = df_imputed['PC1_Car_Specs']
df_final['PC2_Car_Specs'] = df_imputed['PC2_Car_Specs']

# Limpieza final de posibles filas duplicadas creadas por el manejo de índices
df_final.drop_duplicates(inplace=True)

# Guardar el Data Set Limpio para la entrega
df_final.to_csv(OUTPUT_FILE, index=False)

print("\n--- ¡PROCESO FINALIZADO CON ÉXITO! ---")
print(f"Data Set Limpio y Preparado guardado en: {OUTPUT_FILE}")
print(f"Filas finales listas para Streamlit: {len(df_final)}")

# --- 7. ANÁLISIS EXPLORATORIO DE DATOS (EDA) Y VISUALIZACIONES PARA EL INFORME ---

print("\n--- 7. Generando Visualizaciones Clave para el Informe ---")

# Importaciones adicionales necesarias para la visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración básica de Seaborn/Matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==============================================================================
# 7.1. VISUALIZACIÓN DEL PROCESO DE LIMPIEZA (EJEMPLO: ANTES vs. DESPUÉS DE OUTLIERS)
# NOTA: Para el informe, este tipo de gráfico debe hacerse ANTES de ejecutar el
# código de 'Capping con IQR', o se debe cargar el DataFrame original imputado
# ANTES del Capping (df_imputed_pre_capping) para una comparación directa.
# Dado que se ejecutó el Capping, demostraremos el resultado final.
# ==============================================================================

# Gráfico de la distribución de una variable clave (p. ej., 'price')
# para mostrar la distribución después de la imputación y el capping.

plt.figure(figsize=(10, 5))
sns.histplot(df_final['price'], bins=30, kde=True, color='skyblue')
plt.title('Distribución de "price" (Imputado y con Outliers Ajustados)', fontsize=14)
plt.xlabel('Precio (USD)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.tight_layout()
# Puedes guardar el gráfico para el informe si lo deseas
# plt.savefig('distribucion_precio_limpio.png')
# plt.show() # Descomentar para ver en un entorno interactivo


# ==============================================================================
# 7.2. VISUALIZACIÓN DE COMPONENTES PRINCIPALES (PCA)
# Esencial para el informe y el panel de Streamlit
# ==============================================================================

plt.figure(figsize=(12, 8))
# Usamos 'region' para colorear los puntos y ver si PCA separa regiones
scatter = sns.scatterplot(
    x='PC1_Car_Specs',
    y='PC2_Car_Specs',
    hue='region', # Muestra la segmentación por regiones
    data=df_final,
    palette='Spectral',
    alpha=0.7,
    s=70
)

plt.title('PCA - Representación de Especificaciones de Autos Eléctricos por Región', fontsize=16)
plt.xlabel(f'Componente Principal 1 (PC1) - Varianza Explicada: {pca.explained_variance_ratio_[0]:.2%}', fontsize=12)
plt.ylabel(f'Componente Principal 2 (PC2) - Varianza Explicada: {pca.explained_variance_ratio_[1]:.2%}', fontsize=12)
plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# plt.savefig('pca_car_specs.png')
# plt.show() # Descomentar para ver en un entorno interactivo
print("PCA Scatter Plot generado.")

# ==============================================================================
# 7.3. ANÁLISIS DE TENDENCIAS (VENTAS HISTÓRICAS)
# Fundamental para el contexto de ventas
# ==============================================================================

# Agrupar las ventas por año y modo (Cars, Buses, Vans, Trucks)
sales_trend = df_final[df_final['value'].notna()].groupby(['year', 'mode']).agg(
    total_value=('value', 'sum') # 'value' representa EV stock/sales/etc.
).reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=sales_trend,
    x='year',
    y='total_value',
    hue='mode',
    marker='o',
    palette='tab10'
)

plt.title('Tendencia Histórica Agregada de la Métrica "Value" por Modo', fontsize=14)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Valor Agregado (Stock/Ventas, etc.)', fontsize=12)
plt.legend(title='Modo', loc='upper left')
plt.tight_layout()
# plt.savefig('tendencia_ventas_historicas.png')
# plt.show() # Descomentar para ver en un entorno interactivo
print("Gráfico de Tendencia Histórica generado.")

# ==============================================================================
# 7.4. CORRELACIÓN (Para entender la relación entre variables imputadas/ajustadas)
# ==============================================================================

# Matriz de correlación para las variables clave después de la limpieza
correlation_matrix = df_final[IMPUTATION_COLS].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=.5,
    cbar_kws={'label': 'Coeficiente de Correlación'}
)
plt.title('Matriz de Correlación de Variables Numéricas Clave (Post-Limpieza)', fontsize=14)
plt.tight_layout()
# plt.savefig('matriz_correlacion.png')
# plt.show() # Descomentar para ver en un entorno interactivo
print("Matriz de Correlación generada.")

print("\n--- Visualizaciones listas para incluir en el Informe de Limpieza. ---")
print("El siguiente paso sería la implementación del panel en Streamlit.")

# --- FIN DE LA EXTENSIÓN DEL CÓDIGO ---