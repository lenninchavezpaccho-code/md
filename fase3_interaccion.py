import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

# ================================================================================
# CONFIGURACIÓN
# ================================================================================

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Carpetas
CARPETA_FASE1 = 'resultados_fase1_interaccion'
CARPETA_FASE2 = 'resultados_fase2_interaccion'
CARPETA_RESULTADOS = 'resultados_fase3_interaccion'

# Crear carpeta de resultados
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# Configurar logger
doc = open(f'{CARPETA_RESULTADOS}/reporte_fase3_consolidado.txt', 'w', encoding='utf-8')

def log(texto):
    """Escribe en consola y en el archivo de reporte."""
    print(texto)
    doc.write(texto + '\n')

# ================================================================================
# ENCABEZADO
# ================================================================================

log("="*100)
log("FASE 3: VISUALIZACIÓN AVANZADA Y REPORTE EJECUTIVO CONSOLIDADO")
log("Análisis de Interacciones: Volatilidad_SP500 (I1) y Beta_Movil_36M (I2) con COVID-19")
log("="*100)
log(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ================================================================================
# PASO 1: CARGAR DATOS DE TODAS LAS FASES
# ================================================================================

log("="*100)
log("PASO 1: CARGA DE DATOS DE FASES ANTERIORES")
log("="*100)

try:
    # Cargar datos centrados (Fase 2)
    df = pd.read_csv(f'{CARPETA_FASE2}/04_datos_centrados.csv', index_col='Fecha', parse_dates=True)
    log(f"\n✓ Datos centrados cargados desde Fase 2")
    log(f"  Observaciones: {len(df)}")
    log(f"  Período: {df.index.min().strftime('%Y-%m')} a {df.index.max().strftime('%Y-%m')}")
    
    # Cargar VIF ingenuo (Fase 1)
    vif_ingenuo = pd.read_csv(f'{CARPETA_FASE1}/02_vif_ingenuo.csv')
    log(f"\n✓ VIF Ingenuo cargado (Fase 1)")
    
    # Cargar VIF correcto (Fase 2)
    vif_correcto = pd.read_csv(f'{CARPETA_FASE2}/05_vif_correcto.csv')
    log(f"✓ VIF Correcto cargado (Fase 2)")
    
    # Cargar comparación VIF
    comparacion_vif = pd.read_csv(f'{CARPETA_FASE2}/06_comparacion_vif.csv')
    log(f"✓ Comparación VIF cargada (Fase 2)")
    
except FileNotFoundError as e:
    log(f"\n✗ ERROR: No se encontraron archivos de fases anteriores")
    log(f"  Asegúrate de haber ejecutado las Fases 1 y 2 primero.")
    log(f"  Archivo faltante: {e.filename}")
    doc.close()
    exit()
except Exception as e:
    log(f"\n✗ ERROR al cargar datos: {e}")
    doc.close()
    exit()

# ================================================================================
# PASO 2: ANÁLISIS TEMPORAL Y VISUALIZACIONES AVANZADAS
# ================================================================================

log("\n" + "="*100)
log("PASO 2: ANÁLISIS TEMPORAL Y VISUALIZACIONES AVANZADAS")
log("="*100)

# --- 2.1: Gráfico de Series Temporales (Variables Originales) ---
log("\n--- 2.1: Series Temporales de Variables Originales ---")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Volatilidad SP500
axes[0].plot(df.index, df['I1_Volatilidad'], color='#e74c3c', linewidth=2, label='Volatilidad SP500', alpha=0.8)
axes[0].axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Inicio COVID-19')
axes[0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), alpha=0.15, color='red', label='Período COVID')
axes[0].set_ylabel('Volatilidad SP500', fontsize=12, fontweight='bold')
axes[0].set_title('Evolución Temporal de Indicadores de Riesgo Financiero (2013-2024)', fontsize=14, fontweight='bold', pad=15)
axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
axes[0].grid(alpha=0.3, linestyle='--')

# Beta Móvil 36M
axes[1].plot(df.index, df['I2_Beta'], color='#3498db', linewidth=2, label='Beta Móvil 36M (BVL vs SP500)', alpha=0.8)
axes[1].axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', linewidth=2.5, alpha=0.7)
axes[1].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), alpha=0.15, color='red')
axes[1].set_ylabel('Beta Móvil 36M', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=10, framealpha=0.9)
axes[1].grid(alpha=0.3, linestyle='--')

# Dummy COVID
axes[2].fill_between(df.index, 0, df['D_COVID'], color='#e67e22', alpha=0.5, label='D_COVID (Variable Moderadora)')
axes[2].set_ylabel('D_COVID', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Fecha', fontsize=12, fontweight='bold')
axes[2].set_ylim(-0.1, 1.1)
axes[2].set_yticks([0, 1])
axes[2].legend(loc='upper left', fontsize=10, framealpha=0.9)
axes[2].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{CARPETA_RESULTADOS}/09_series_temporales_originales.png', dpi=300, bbox_inches='tight')
plt.close()

log("✓ Gráfico de series temporales guardado: 09_series_temporales_originales.png")

# --- 2.2: Gráfico de Variables Centradas e Interacciones ---
log("\n--- 2.2: Variables Centradas e Interacciones ---")

fig, axes = plt.subplots(2, 2, figsize=(18, 11))

# Volatilidad Centrada
axes[0, 0].plot(df.index, df['I1_Volatilidad_c'], color='#9b59b6', linewidth=2, alpha=0.8)
axes[0, 0].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[0, 0].axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), alpha=0.15, color='red')
axes[0, 0].set_title('I1_Volatilidad_c (Centrada)', fontsize=13, fontweight='bold', pad=10)
axes[0, 0].set_ylabel('Volatilidad Centrada', fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3, linestyle='--')

# Beta Centrada
axes[0, 1].plot(df.index, df['I2_Beta_c'], color='#1abc9c', linewidth=2, alpha=0.8)
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[0, 1].axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), alpha=0.15, color='red')
axes[0, 1].set_title('I2_Beta_c (Centrada)', fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_ylabel('Beta Centrada', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3, linestyle='--')

# Interacción Vol x COVID
axes[1, 0].plot(df.index, df['Int_Vol_COVID'], color='#e74c3c', linewidth=2, label='Int_Vol_COVID', alpha=0.8)
axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[1, 0].fill_between(df.index, 0, df['Int_Vol_COVID'], where=(df['D_COVID'] == 1), 
                        alpha=0.3, color='#e74c3c', label='Período COVID Activo')
axes[1, 0].set_title('Int_Vol_COVID (Interacción 1)', fontsize=13, fontweight='bold', pad=10)
axes[1, 0].set_xlabel('Fecha', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Interacción', fontsize=11, fontweight='bold')
axes[1, 0].legend(loc='best', fontsize=9, framealpha=0.9)
axes[1, 0].grid(alpha=0.3, linestyle='--')

# Interacción Beta x COVID
axes[1, 1].plot(df.index, df['Int_Beta_COVID'], color='#3498db', linewidth=2, label='Int_Beta_COVID', alpha=0.8)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[1, 1].fill_between(df.index, 0, df['Int_Beta_COVID'], where=(df['D_COVID'] == 1), 
                        alpha=0.3, color='#3498db', label='Período COVID Activo')
axes[1, 1].set_title('Int_Beta_COVID (Interacción 2)', fontsize=13, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Fecha', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Interacción', fontsize=11, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=9, framealpha=0.9)
axes[1, 1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{CARPETA_RESULTADOS}/10_variables_centradas_interacciones.png', dpi=300, bbox_inches='tight')
plt.close()

log("✓ Gráfico de variables centradas e interacciones guardado: 10_variables_centradas_interacciones.png")

# --- 2.3: Análisis Comparativo Pre/Post COVID ---
log("\n--- 2.3: Análisis Comparativo Pre/Post COVID ---")

df_pre_covid = df[df['D_COVID'] == 0]
df_post_covid = df[df['D_COVID'] == 1]

# Estadísticas comparativas
stats_comparacion = pd.DataFrame({
    'Variable': ['I1_Volatilidad', 'I2_Beta'],
    'Media_Pre_COVID': [df_pre_covid['I1_Volatilidad'].mean(), df_pre_covid['I2_Beta'].mean()],
    'Media_Post_COVID': [df_post_covid['I1_Volatilidad'].mean(), df_post_covid['I2_Beta'].mean()],
    'Desv_Pre_COVID': [df_pre_covid['I1_Volatilidad'].std(), df_pre_covid['I2_Beta'].std()],
    'Desv_Post_COVID': [df_post_covid['I1_Volatilidad'].std(), df_post_covid['I2_Beta'].std()]
})

stats_comparacion['Cambio_Media_%'] = ((stats_comparacion['Media_Post_COVID'] - stats_comparacion['Media_Pre_COVID']) 
                                       / stats_comparacion['Media_Pre_COVID'] * 100)

stats_comparacion['Cambio_Desv_%'] = ((stats_comparacion['Desv_Post_COVID'] - stats_comparacion['Desv_Pre_COVID']) 
                                       / stats_comparacion['Desv_Pre_COVID'] * 100)

log("\n--- Estadísticas Comparativas Pre/Post COVID ---")
log(stats_comparacion.to_string(index=False, float_format='{:.4f}'.format))

stats_comparacion.to_csv(f'{CARPETA_RESULTADOS}/11_comparacion_pre_post_covid.csv', index=False)

# Visualización comparativa (Boxplots mejorados)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Volatilidad
data_vol = pd.DataFrame({
    'Período': ['Pre-COVID']*len(df_pre_covid) + ['Post-COVID']*len(df_post_covid),
    'Volatilidad': list(df_pre_covid['I1_Volatilidad']) + list(df_post_covid['I1_Volatilidad'])
})
bp1 = sns.boxplot(x='Período', y='Volatilidad', data=data_vol, ax=axes[0], palette=['#3498db', '#e74c3c'], width=0.6)
axes[0].set_title('Distribución de Volatilidad SP500: Pre vs Post COVID', fontsize=13, fontweight='bold', pad=12)
axes[0].set_ylabel('Volatilidad SP500', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Período', fontsize=11, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Añadir estadísticas al gráfico
y_pos = axes[0].get_ylim()[1] * 0.95
axes[0].text(0.5, y_pos, f'Cambio: {stats_comparacion.loc[0, "Cambio_Media_%"]:+.1f}%', 
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Beta
data_beta = pd.DataFrame({
    'Período': ['Pre-COVID']*len(df_pre_covid) + ['Post-COVID']*len(df_post_covid),
    'Beta': list(df_pre_covid['I2_Beta']) + list(df_post_covid['I2_Beta'])
})
bp2 = sns.boxplot(x='Período', y='Beta', data=data_beta, ax=axes[1], palette=['#3498db', '#e74c3c'], width=0.6)
axes[1].set_title('Distribución de Beta Móvil 36M: Pre vs Post COVID', fontsize=13, fontweight='bold', pad=12)
axes[1].set_ylabel('Beta Móvil 36M (BVL vs SP500)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Período', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Añadir estadísticas al gráfico
y_pos = axes[1].get_ylim()[1] * 0.95
axes[1].text(0.5, y_pos, f'Cambio: {stats_comparacion.loc[1, "Cambio_Media_%"]:+.1f}%', 
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{CARPETA_RESULTADOS}/11_boxplot_pre_post_covid.png', dpi=300, bbox_inches='tight')
plt.close()

log("\n✓ Análisis comparativo Pre/Post COVID guardado:")
log("  11_comparacion_pre_post_covid.csv")
log("  11_boxplot_pre_post_covid.png")

# --- 2.4: Scatter Plots con Líneas de Regresión ---
log("\n--- 2.4: Scatter Plots - Relación con Interacciones ---")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Volatilidad vs Interacción
axes[0].scatter(df[df['D_COVID']==0]['I1_Volatilidad_c'], 
               df[df['D_COVID']==0]['Int_Vol_COVID'], 
               alpha=0.6, s=60, color='#3498db', label='Pre-COVID', edgecolors='black', linewidth=0.7)
axes[0].scatter(df[df['D_COVID']==1]['I1_Volatilidad_c'], 
               df[df['D_COVID']==1]['Int_Vol_COVID'], 
               alpha=0.85, s=90, color='#e74c3c', label='Post-COVID', edgecolors='black', linewidth=0.7, marker='s')
axes[0].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[0].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[0].set_xlabel('I1_Volatilidad_c (Centrada)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Int_Vol_COVID', fontsize=12, fontweight='bold')
axes[0].set_title('Relación: Volatilidad Centrada vs Interacción COVID', fontsize=13, fontweight='bold', pad=12)
axes[0].legend(loc='best', fontsize=11, framealpha=0.9)
axes[0].grid(alpha=0.3, linestyle='--')

# Beta vs Interacción
axes[1].scatter(df[df['D_COVID']==0]['I2_Beta_c'], 
               df[df['D_COVID']==0]['Int_Beta_COVID'], 
               alpha=0.6, s=60, color='#3498db', label='Pre-COVID', edgecolors='black', linewidth=0.7)
axes[1].scatter(df[df['D_COVID']==1]['I2_Beta_c'], 
               df[df['D_COVID']==1]['Int_Beta_COVID'], 
               alpha=0.85, s=90, color='#e74c3c', label='Post-COVID', edgecolors='black', linewidth=0.7, marker='s')
axes[1].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[1].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
axes[1].set_xlabel('I2_Beta_c (Centrada)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Int_Beta_COVID', fontsize=12, fontweight='bold')
axes[1].set_title('Relación: Beta Centrada vs Interacción COVID', fontsize=13, fontweight='bold', pad=12)
axes[1].legend(loc='best', fontsize=11, framealpha=0.9)
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{CARPETA_RESULTADOS}/12_scatter_interacciones.png', dpi=300, bbox_inches='tight')
plt.close()

log("\n✓ Scatter plots guardados: 12_scatter_interacciones.png")

# --- 2.5: Histogramas de Distribuciones ---
log("\n--- 2.5: Histogramas de Distribuciones ---")

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# Volatilidad Original
axes[0, 0].hist(df['I1_Volatilidad'], bins=35, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.2)
axes[0, 0].axvline(df['I1_Volatilidad'].mean(), color='darkred', linestyle='--', linewidth=2.5, 
                   label=f"Media: {df['I1_Volatilidad'].mean():.4f}")
axes[0, 0].set_title('Distribución I1_Volatilidad (Original)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Volatilidad SP500', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3, linestyle='--')

# Beta Original
axes[0, 1].hist(df['I2_Beta'], bins=35, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.2)
axes[0, 1].axvline(df['I2_Beta'].mean(), color='darkblue', linestyle='--', linewidth=2.5, 
                   label=f"Media: {df['I2_Beta'].mean():.4f}")
axes[0, 1].set_title('Distribución I2_Beta (Original)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Beta Móvil 36M', fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3, linestyle='--')

# D_COVID
covid_counts = df['D_COVID'].value_counts().sort_index()
axes[0, 2].bar([0, 1], covid_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.2, width=0.6)
axes[0, 2].set_title('Distribución D_COVID (Moderadora)', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('D_COVID', fontsize=10, fontweight='bold')
axes[0, 2].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
axes[0, 2].set_xticks([0, 1])
axes[0, 2].set_xticklabels(['Pre-COVID (0)', 'Post-COVID (1)'], fontsize=9)
for i, v in enumerate(covid_counts.values):
    axes[0, 2].text(i, v + max(covid_counts.values)*0.02, str(v), ha='center', fontsize=10, fontweight='bold')
axes[0, 2].grid(alpha=0.3, linestyle='--', axis='y')

# Volatilidad Centrada
axes[1, 0].hist(df['I1_Volatilidad_c'], bins=35, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.2)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2.5, label='Media: 0')
axes[1, 0].set_title('Distribución I1_Volatilidad_c (Centrada)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Volatilidad Centrada', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3, linestyle='--')

# Beta Centrada
axes[1, 1].hist(df['I2_Beta_c'], bins=35, color='#1abc9c', alpha=0.7, edgecolor='black', linewidth=1.2)
axes[1, 1].axvline(0, color='blue', linestyle='--', linewidth=2.5, label='Media: 0')
axes[1, 1].set_title('Distribución I2_Beta_c (Centrada)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Beta Centrada', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3, linestyle='--')

# Interacciones superpuestas
axes[1, 2].hist(df['Int_Vol_COVID'], bins=25, color='#e74c3c', alpha=0.5, edgecolor='black', 
                linewidth=1.2, label='Int_Vol_COVID')
axes[1, 2].hist(df['Int_Beta_COVID'], bins=25, color='#3498db', alpha=0.5, edgecolor='black', 
                linewidth=1.2, label='Int_Beta_COVID')
axes[1, 2].set_title('Distribución de Interacciones', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Valor Interacción', fontsize=10, fontweight='bold')
axes[1, 2].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
axes[1, 2].legend(fontsize=9, framealpha=0.9)
axes[1, 2].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{CARPETA_RESULTADOS}/13_histogramas_distribuciones.png', dpi=300, bbox_inches='tight')
plt.close()

log("\n✓ Histogramas guardados: 13_histogramas_distribuciones.png")

# --- 2.6: Análisis de Correlación ---
log("\n--- 2.6: Análisis de Correlación entre Variables ---")

# Calcular correlaciones
vars_analisis = ['I1_Volatilidad_c', 'I2_Beta_c', 'D_COVID', 'Int_Vol_COVID', 'Int_Beta_COVID']
corr_matrix = df[vars_analisis].corr()

log("\n--- Matriz de Correlación (Variables Centradas) ---")
log(corr_matrix.to_string(float_format='{:.4f}'.format))

# Heatmap de correlaciones
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
            vmin=-1, vmax=1, square=True, linewidths=2, cbar_kws={"shrink": 0.8}, 
            mask=mask, ax=ax, annot_kws={"size": 11, "weight": "bold"})
ax.set_title('Matriz de Correlación - Variables Centradas e Interacciones\n✅ Modelo Correcto (Post-Centrado)', 
             fontweight='bold', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(f'{CARPETA_RESULTADOS}/13b_heatmap_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()

log("\n✓ Heatmap de correlación guardado: 13b_heatmap_correlacion.png")

# ================================================================================
# PASO 3: DASHBOARD INTEGRADO (RESUMEN VISUAL)
# ================================================================================

log("\n" + "="*100)
log("PASO 3: DASHBOARD INTEGRADO - RESUMEN EJECUTIVO VISUAL")
log("="*100)

fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.30)

# Panel 1: Comparación VIF (Barras) - Más grande
ax1 = fig.add_subplot(gs[0:2, :2])
x = np.arange(len(comparacion_vif))
width = 0.35
bars1 = ax1.bar(x - width/2, comparacion_vif['VIF_Antes_Sin_Centrar'], width, 
               label='VIF Antes (Sin Centrar)', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, comparacion_vif['VIF_Después_Con_Centrar'], width, 
               label='VIF Después (Con Centrar)', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)

# Añadir valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('VIF', fontweight='bold', fontsize=12)
ax1.set_title('A) Comparación VIF: Solución de Multicolinealidad No Esencial', 
              fontweight='bold', fontsize=14, pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(comparacion_vif['Variable'], rotation=35, ha='right', fontsize=10)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Crítico (VIF=10)')
ax1.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Moderado (VIF=5)')
ax1.grid(axis='y', alpha=0.4, linestyle='--')
ax1.set_ylim(0, max(comparacion_vif['VIF_Antes_Sin_Centrar'].max() * 1.15, 15))

# Panel 2: Métricas clave
ax2 = fig.add_subplot(gs[0:2, 2])
ax2.axis('off')

# Calcular reducción promedio
reduccion_promedio = comparacion_vif['Reducción_%'].mean()
vif_max_antes = comparacion_vif['VIF_Antes_Sin_Centrar'].max()
vif_max_despues = comparacion_vif['VIF_Después_Con_Centrar'].max()
vars_problema_antes = len(comparacion_vif[comparacion_vif['VIF_Antes_Sin_Centrar'] > 10])
vars_problema_despues = len(comparacion_vif[comparacion_vif['VIF_Después_Con_Centrar'] > 10])

metricas_texto = f"""
MÉTRICAS CLAVE DEL ANÁLISIS

📊 REDUCCIÓN VIF:
   Promedio: {reduccion_promedio:.1f}%
   
     D_COVID: 
     {comparacion_vif[comparacion_vif['Variable']=='D_COVID']['Reducción_%'].values[0]:.1f}%
   
     Int_Beta_COVID: 
     {comparacion_vif[comparacion_vif['Variable']=='Int_Beta_COVID']['Reducción_%'].values[0]:.1f}%
   
     Int_Vol_COVID: 
     {comparacion_vif[comparacion_vif['Variable']=='Int_Vol_COVID']['Reducción_%'].values[0]:.1f}%

🎯 MULTICOLINEALIDAD:
   Antes:  {vars_problema_antes} var(s) VIF>10
   Después: {vars_problema_despues} var(s) VIF>10
   
   VIF Máximo:
     Antes:  {vif_max_antes:.2f}
     Después: {vif_max_despues:.2f}

📅 PERÍODO:
   {df.index.min().strftime('%Y-%m')} a {df.index.max().strftime('%Y-%m')}
   N = {len(df)} meses
   
   COVID-19:
     Pre: {len(df_pre_covid)} meses
     Post: {len(df_post_covid)} meses

✅ STATUS:
   Solución EXITOSA
   Listo para regresión
"""

ax2.text(0.05, 0.98, metricas_texto, fontsize=10, verticalalignment='top', 
         family='monospace', bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8, edgecolor='black', linewidth=2))

# Panel 3: Serie temporal Volatilidad con anotaciones
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(df.index, df['I1_Volatilidad'], color='#e74c3c', linewidth=2, label='Volatilidad SP500', alpha=0.85)
ax3.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Inicio COVID-19')
ax3.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), alpha=0.15, color='red', label='Período COVID')

# Añadir estadísticas
media_pre = df_pre_covid['I1_Volatilidad'].mean()
media_post = df_post_covid['I1_Volatilidad'].mean()
ax3.axhline(media_pre, color='blue', linestyle=':', linewidth=2, alpha=0.6, label=f'Media Pre-COVID: {media_pre:.4f}')
ax3.axhline(media_post, color='darkred', linestyle=':', linewidth=2, alpha=0.6, label=f'Media Post-COVID: {media_post:.4f}')

ax3.set_ylabel('Volatilidad SP500', fontweight='bold', fontsize=11)
ax3.set_title('B) Evolución Temporal: Volatilidad SP500 (Indicador I1)', fontweight='bold', fontsize=13, pad=10)
ax3.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
ax3.grid(alpha=0.3, linestyle='--')

# Panel 4: Serie temporal Beta con anotaciones
ax4 = fig.add_subplot(gs[3, :])
ax4.plot(df.index, df['I2_Beta'], color='#3498db', linewidth=2, label='Beta Móvil 36M', alpha=0.85)
ax4.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax4.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), alpha=0.15, color='red')

# Añadir estadísticas
media_pre_beta = df_pre_covid['I2_Beta'].mean()
media_post_beta = df_post_covid['I2_Beta'].mean()
ax4.axhline(media_pre_beta, color='blue', linestyle=':', linewidth=2, alpha=0.6, label=f'Media Pre-COVID: {media_pre_beta:.4f}')
ax4.axhline(media_post_beta, color='darkred', linestyle=':', linewidth=2, alpha=0.6, label=f'Media Post-COVID: {media_post_beta:.4f}')

ax4.set_ylabel('Beta Móvil 36M', fontweight='bold', fontsize=11)
ax4.set_xlabel('Fecha', fontweight='bold', fontsize=11)
ax4.set_title('C) Evolución Temporal: Beta Móvil 36M - BVL vs SP500 (Indicador I2)', fontweight='bold', fontsize=13, pad=10)
ax4.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
ax4.grid(alpha=0.3, linestyle='--')

plt.suptitle('DASHBOARD EJECUTIVO - Análisis de Interacciones con Variable Moderadora COVID-19\nIndicadores: Volatilidad_SP500 (I1) y Beta_Movil_36M (I2)', 
             fontsize=16, fontweight='bold', y=0.998)
plt.savefig(f'{CARPETA_RESULTADOS}/14_dashboard_ejecutivo.png', dpi=300, bbox_inches='tight')
plt.close()

log("\n✓ Dashboard ejecutivo guardado: 14_dashboard_ejecutivo.png")

# ================================================================================
# PASO 4: DOCUMENTACIÓN METODOLÓGICA PARA TESIS
# ================================================================================

log("\n" + "="*100)
log("PASO 4: DOCUMENTACIÓN METODOLÓGICA PARA TESIS")
log("="*100)

doc_metodologia = open(f'{CARPETA_RESULTADOS}/15_documentacion_metodologica.txt', 'w', encoding='utf-8')

metodologia = f"""
================================================================================
DOCUMENTACIÓN METODOLÓGICA - ANÁLISIS DE INTERACCIONES
Análisis de Moderación con Variable Dummy COVID-19
Preparado para: Tesis de Investigación
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. CONTEXTO DEL ANÁLISIS
================================================================================

OBJETIVO GENERAL:
  Evaluar el efecto moderador de la pandemia COVID-19 (D_COVID) en la relación
  entre dos indicadores de riesgo financiero del mercado peruano:
  
    I1_Volatilidad_SP500: Volatilidad del índice S&P500 (ventana móvil 36 meses)
    Representa el riesgo de mercado global que afecta a mercados emergentes
    
    I2_Beta_Movil_36M: Beta del mercado BVL respecto al S&P500 (ventana móvil 36M)
    Mide la sensibilidad del mercado peruano ante movimientos del mercado global

PERÍODO DE ANÁLISIS:
    Inicio: {df.index.min().strftime('%Y-%m-%d')}
    Fin: {df.index.max().strftime('%Y-%m-%d')}
    Total observaciones: {len(df)} meses ({(len(df)/12):.1f} años)
  
    Período Pre-COVID: {len(df_pre_covid)} meses ({(len(df_pre_covid)/len(df)*100):.1f}%)
    Comprende: {df_pre_covid.index.min().strftime('%Y-%m')} a {df_pre_covid.index.max().strftime('%Y-%m')}
    
    Período Post-COVID: {len(df_post_covid)} meses ({(len(df_post_covid)/len(df)*100):.1f}%)
    Comprende: {df_post_covid.index.min().strftime('%Y-%m')} a {df_post_covid.index.max().strftime('%Y-%m')}

VARIABLE MODERADORA:
    D_COVID: Variable dummy binaria (0/1)
    Valor 1: Marzo 2020 - Diciembre 2021 (período pandemia activa)
    Valor 0: Resto del período muestral
    Justificación: Captura el período de mayor impacto económico de la pandemia,
    incluyendo lockdowns, volatilidad extrema y políticas monetarias expansivas

2. FUNDAMENTACIÓN TEÓRICA
================================================================================

MARCO DE REFERENCIA:
  Aiken, L. S., & West, S. G. (1991). Multiple Regression: Testing and 
  Interpreting Interactions. Sage Publications.

PROBLEMA METODOLÓGICO IDENTIFICADO:
  
  Al crear términos de interacción (X × Z) directamente con variables no 
  centradas, se genera "multicolinealidad no esencial" - una correlación 
  artificial entre las variables principales y sus interacciones.
  
  Esta multicolinealidad es un ARTEFACTO MATEMÁTICO, no un problema real 
  de los datos. Se manifiesta en valores VIF inflados (>10).

SOLUCIÓN METODOLÓGICA:
  
  Centrado de variables continuas ANTES de crear términos de interacción:
  
  PASO 1: Centrar variables continuas
    I1_Volatilidad_c = I1_Volatilidad - mean(I1_Volatilidad)
    I2_Beta_c = I2_Beta - mean(I2_Beta)
  
  PASO 2: NO centrar la variable dummy
    D_COVID permanece con valores 0/1 (mantiene interpretabilidad)
  
  PASO 3: Crear interacciones con variables centradas
    Int_Vol_COVID = I1_Volatilidad_c × D_COVID
    Int_Beta_COVID = I2_Beta_c × D_COVID

BENEFICIOS DEL CENTRADO:
  
  ✓ Elimina multicolinealidad no esencial (reducción promedio VIF: {reduccion_promedio:.1f}%)
  ✓ Preserva la interpretabilidad de coeficientes
  ✓ Mejora estabilidad numérica del modelo
  ✓ No afecta R², F-test, ni significancia estadística de interacciones

3. RESULTADOS CLAVE
================================================================================

A. REDUCCIÓN DE MULTICOLINEALIDAD

{comparacion_vif.to_string(index=False)}

B. ESTADÍSTICAS DESCRIPTIVAS

Período Pre-COVID (n={len(df_pre_covid)} meses):
  
  I1_Volatilidad_SP500:
    Media:        {df_pre_covid['I1_Volatilidad'].mean():.6f}
    Desv. Est.:   {df_pre_covid['I1_Volatilidad'].std():.6f}
    Mínimo:       {df_pre_covid['I1_Volatilidad'].min():.6f}
    Máximo:       {df_pre_covid['I1_Volatilidad'].max():.6f}
    
  I2_Beta_Movil_36M:
    Media:        {df_pre_covid['I2_Beta'].mean():.6f}
    Desv. Est.:   {df_pre_covid['I2_Beta'].std():.6f}
    Mínimo:       {df_pre_covid['I2_Beta'].min():.6f}
    Máximo:       {df_pre_covid['I2_Beta'].max():.6f}

Período Post-COVID (n={len(df_post_covid)} meses):
  
  I1_Volatilidad_SP500:
    Media:        {df_post_covid['I1_Volatilidad'].mean():.6f}
    Desv. Est.:   {df_post_covid['I1_Volatilidad'].std():.6f}
    Mínimo:       {df_post_covid['I1_Volatilidad'].min():.6f}
    Máximo:       {df_post_covid['I1_Volatilidad'].max():.6f}
    
  I2_Beta_Movil_36M:
    Media:        {df_post_covid['I2_Beta'].mean():.6f}
    Desv. Est.:   {df_post_covid['I2_Beta'].std():.6f}
    Mínimo:       {df_post_covid['I2_Beta'].min():.6f}
    Máximo:       {df_post_covid['I2_Beta'].max():.6f}

C. CAMBIOS PRE/POST COVID:

I1_Volatilidad:
     Cambio Media: {stats_comparacion.loc[0, 'Cambio_Media_%']:+.2f}%
     Cambio Desv.: {stats_comparacion.loc[0, 'Cambio_Desv_%']:+.2f}%

I2_Beta:
     Cambio Media: {stats_comparacion.loc[1, 'Cambio_Media_%']:+.2f}%
     Cambio Desv.: {stats_comparacion.loc[1, 'Cambio_Desv_%']:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 ARCHIVOS GENERADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DATOS Y ANÁLISIS:
     08_dataset_final_regresion.xlsx ⭐ (USAR PARA REGRESIÓN)
     11_comparacion_pre_post_covid.csv
     06_comparacion_vif.csv

📈 VISUALIZACIONES:
     14_dashboard_ejecutivo.png ⭐ (RESUMEN VISUAL)
     09_series_temporales_originales.png
     10_variables_centradas_interacciones.png
     11_boxplot_pre_post_covid.png
     12_scatter_interacciones.png
     13_histogramas_distribuciones.png
     13b_heatmap_correlacion.png

📄 DOCUMENTACIÓN:
     15_documentacion_metodologica.txt ⭐ (PARA TESIS)
     16_resumen_ejecutivo_final.txt (ESTE ARCHIVO)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ CONCLUSIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALIDACIÓN METODOLÓGICA:
   ✓ Multicolinealidad no esencial eliminada
   ✓ Dataset preparado para regresión moderada
   ✓ Visualizaciones completas para tesis
   ✓ Documentación metodológica lista

PRÓXIMO PASO:
   Estimar modelo de regresión con el archivo:
   08_dataset_final_regresion.xlsx

VARIABLES DEL MODELO:
   Y ~ I1_Volatilidad_c + I2_Beta_c + D_COVID + 
       Int_Vol_COVID + Int_Beta_COVID

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Guardar resumen ejecutivo
with open(f'{CARPETA_RESULTADOS}/16_resumen_ejecutivo_final.txt', 'w', encoding='utf-8') as f:
    f.write(resumen_ejecutivo)

log(resumen_ejecutivo)

log("\n✓ Resumen ejecutivo guardado: 16_resumen_ejecutivo_final.txt")

# ================================================================================
# FINALIZACIÓN
# ================================================================================

doc.close()

print("\n" + "="*100)
print("✅✅✅ FASE 3 COMPLETADA EXITOSAMENTE ✅✅✅")
print("="*100)
print("Se ha generado un análisis completo con visualizaciones avanzadas.")
print(f"\nRevisa los resultados en la carpeta: {CARPETA_RESULTADOS}/")
print("\n📋 ARCHIVOS PRINCIPALES GENERADOS:")
print("\n📊 VISUALIZACIONES:")
print("    09_series_temporales_originales.png")
print("    10_variables_centradas_interacciones.png")
print("    11_boxplot_pre_post_covid.png")
print("    12_scatter_interacciones.png")
print("    13_histogramas_distribuciones.png")
print("    13b_heatmap_correlacion.png")
print("    14_dashboard_ejecutivo.png ⭐ (RESUMEN INTEGRADO)")
print("\n📄 DOCUMENTACIÓN:")
print("    15_documentacion_metodologica.txt ⭐ (PARA METODOLOGÍA DE TESIS)")
print("    16_resumen_ejecutivo_final.txt ⭐ (RESUMEN CONSOLIDADO)")
print("    reporte_fase3_consolidado.txt (REPORTE TÉCNICO COMPLETO)")
print("\n📈 DATOS:")
print("    11_comparacion_pre_post_covid.csv")
print("\n🎯 DATASET PARA REGRESIÓN:")
print("    ../resultados_fase2_interaccion/08_dataset_final_regresion.xlsx")
print("\n💡 PRÓXIMO PASO:")
print("  Usar el dataset final para estimar tu modelo de regresión moderada")
print("  en Python (statsmodels), R, Stata o SPSS según tu preferencia.")
print("\n🎓 PARA TU TESIS:")
print("    Usa 14_dashboard_ejecutivo.png en presentaciones")
print("    Copia 15_documentacion_metodologica.txt a tu capítulo de metodología")
print("    Incluye los gráficos 09, 11, 12, 13 en resultados")
print("="*100)IOS PRE/POST COVID

{stats_comparacion.to_string(index=False)}

Interpretación Económica:
  
    Volatilidad SP500: {'Incremento' if stats_comparacion.loc[0, 'Cambio_Media_%'] > 0 else 'Descenso'} del {abs(stats_comparacion.loc[0, 'Cambio_Media_%']):.1f}% durante COVID-19
    Refleja la mayor incertidumbre de mercados globales durante la pandemia.
    
    Beta BVL-SP500: {'Incremento' if stats_comparacion.loc[1, 'Cambio_Media_%'] > 0 else 'Descenso'} del {abs(stats_comparacion.loc[1, 'Cambio_Media_%']):.1f}% durante COVID-19
    {'Mayor' if stats_comparacion.loc[1, 'Cambio_Media_%'] > 0 else 'Menor'} sensibilidad del mercado peruano ante movimientos del SP500.

D. CORRELACIONES (Modelo Correcto)

{corr_matrix.to_string(float_format='{:.4f}'.format)}

4. ECUACIÓN DEL MODELO PROPUESTO
================================================================================

Modelo de Regresión Moderada (especificación completa):

    Y = β₀ + β₁(I1_Volatilidad_c) + β₂(I2_Beta_c) + β₃(D_COVID)
        + β₄(Int_Vol_COVID) + β₅(Int_Beta_COVID) + ε

Donde:
  Y = Variable dependiente (a especificar según tu tesis)
  
  β₁ = Efecto principal de Volatilidad SP500 (centrada)
  β₂ = Efecto principal de Beta (centrada)
  β₃ = Efecto principal de COVID
  β₄ = Efecto de interacción Volatilidad × COVID
  β₅ = Efecto de interacción Beta × COVID

5. ARCHIVOS GENERADOS
================================================================================

FASE 1 (Diagnóstico):
    02_vif_ingenuo.csv
    03_heatmap_ingenuo.png

FASE 2 (Solución):
    04_datos_centrados.csv
    05_vif_correcto.csv
    06_comparacion_vif.csv
    08_dataset_final_regresion.xlsx ⭐ (USAR ESTE)

FASE 3 (Visualización):
    09_series_temporales_originales.png
    10_variables_centradas_interacciones.png
    11_comparacion_pre_post_covid.csv
    11_boxplot_pre_post_covid.png
    12_scatter_interacciones.png
    13_histogramas_distribuciones.png
    13b_heatmap_correlacion.png
    14_dashboard_ejecutivo.png ⭐
    15_documentacion_metodologica.txt ⭐ (ESTE ARCHIVO)

6. RECOMENDACIONES PARA LA TESIS
================================================================================

SECCIÓN DE METODOLOGÍA:
  
    Marco teórico: Citar Aiken & West (1991)
    Justificación del centrado
    Tabla comparativa de VIF (archivo 06)
  
SECCIÓN DE RESULTADOS:
  
    Tabla: Estadísticas descriptivas (archivo 11)
    Figura: Series temporales (archivo 09)
    Figura: Boxplots Pre/Post COVID (archivo 11)
    Figura: Dashboard ejecutivo (archivo 14)

7. PRÓXIMOS PASOS
================================================================================

✅ COMPLETADO:
  [X] Diagnóstico de multicolinealidad
  [X] Solución mediante centrado
  [X] Visualizaciones avanzadas
  [X] Documentación metodológica

⭐ PENDIENTE:
  [ ] Especificar variable dependiente (Y)
  [ ] Estimar modelo de regresión moderada
  [ ] Verificar supuestos del modelo
  [ ] Análisis de efectos condicionales
  [ ] Redacción de tesis

================================================================================
FIN DE LA DOCUMENTACIÓN METODOLÓGICA
================================================================================
"""

doc_metodologia.write(metodologia)
doc_metodologia.close()

log("\n✓ Documentación metodológica guardada: 15_documentacion_metodologica.txt")

# ================================================================================
# PASO 5: RESUMEN EJECUTIVO FINAL
# ================================================================================

log("\n" + "="*100)
log("PASO 5: RESUMEN EJECUTIVO CONSOLIDADO (TODAS LAS FASES)")
log("="*100)

resumen_ejecutivo = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     RESUMEN EJECUTIVO CONSOLIDADO                            ║
║              ANÁLISIS DE INTERACCIONES CON VARIABLE MODERADORA               ║
║                          (FASES 1, 2 Y 3)                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 PERÍODO: {df.index.min().strftime('%Y-%m')} a {df.index.max().strftime('%Y-%m')} ({len(df)} meses)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 FASE 1: DIAGNÓSTICO DEL PROBLEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 PROBLEMA DETECTADO:
   Variables con VIF > 10 (multicolinealidad severa):
     D_COVID: {vif_ingenuo[vif_ingenuo['Variable']=='D_COVID']['VIF'].values[0]:.2f}
     Int_Beta_COVID: {vif_ingenuo[vif_ingenuo['Variable']=='Int_Beta_COVID_ingenuo']['VIF'].values[0]:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ FASE 2: SOLUCIÓN MEDIANTE CENTRADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METODOLOGÍA: Centrado Pre-Interacción (Aiken & West, 1991)

🟢 RESULTADOS:
     Reducción promedio VIF: {reduccion_promedio:.1f}%
     Variables con VIF > 10: {vars_problema_antes} → {vars_problema_despues}
     VIF máximo: {vif_max_antes:.2f} → {vif_max_despues:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 FASE 3: HALLAZGOS PRINCIPALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAMB