"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO FASE 3.2: ESTIMACIÓN DE MODELOS FINALES
═══════════════════════════════════════════════════════════════════════════════
Objetivos:
1. Estimar Panel 1 (Aportes) - Modelo RESTRINGIDO
2. Estimar Panel 2 (Reasignación) - Modelo RESTRINGIDO + Desagregado por Fondo
3. Guardar resultados en formato publicable
═══════════════════════════════════════════════════════════════════════════════
"""

# Importar configuración
try:
    from modulo_0_config import (CONFIG, OUTPUT_DIR, df_p1, df_p2,
                                  ctrl_c_p1, ctrl_c_p2, mes_p1, mes_p2)
    print("✅ Configuración importada desde Módulo 0")
except ImportError:
    print("⚠️  Ejecuta primero: modulo_0_config.py")
    raise

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# ==========================================
# FUNCIÓN: ESTIMAR MODELO RESTRINGIDO
# ==========================================

def estimar_panel_restringido(df, y_col, x_cols, entity_col, time_col, 
                               nombre_panel="Panel", guardar=True):
    """
    Estima modelo de panel con Efectos Fijos (modelo restringido).
    
    Especificación: VD ~ IVs + Controles_centrados + Dummies_Mes + FE_entidad
    
    Parameters:
        df (DataFrame): DataFrame sin MultiIndex
        y_col (str): Variable dependiente
        x_cols (list): Variables independientes
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        nombre_panel (str): Nombre para archivos
        guardar (bool): Si guardar resultados
    
    Returns:
        dict: Resultados del modelo con estadísticos clave
    """
    print(f"\n{'='*70}")
    print(f"ESTIMACIÓN MODELO RESTRINGIDO - {nombre_panel}")
    print(f"{'='*70}")
    
    # Preparar datos
    df_modelo = df.dropna(subset=[y_col] + x_cols)
    df_modelo = df_modelo.set_index([entity_col, time_col])
    
    y = df_modelo[y_col]
    X = df_modelo[x_cols]
    
    print(f"\n📊 Dimensiones del modelo:")
    print(f"   • Observaciones: {len(y)}")
    print(f"   • Entidades: {y.index.get_level_values(0).nunique()}")
    print(f"   • Períodos: {y.index.get_level_values(1).nunique()}")
    print(f"   • Variables X: {len(x_cols)}")
    
    # Estimar modelo
    print(f"\n🔄 Estimando PanelOLS con Efectos Fijos...")
    modelo = PanelOLS(y, X, entity_effects=True).fit(cov_type='robust')
    
    # Mostrar resumen
    print(f"\n{modelo.summary}")
    
    # Extraer resultados clave
    resultados = {
        'modelo': modelo,
        'n_obs': len(y),
        'n_entidades': y.index.get_level_values(0).nunique(),
        'n_periodos': y.index.get_level_values(1).nunique(),
        'r2': modelo.rsquared,
        'r2_within': modelo.rsquared_within,
        'r2_between': modelo.rsquared_between,
        'f_stat': modelo.f_statistic.stat,
        'f_pval': modelo.f_statistic.pval,
        'coefs': modelo.params,
        'se': modelo.std_errors,
        'tstats': modelo.tstats,
        'pvals': modelo.pvalues,
        'residuos': modelo.resids
    }
    
    # Guardar resultados
    if guardar:
        nombre_archivo = nombre_panel.lower().replace(' ', '_').replace(':', '')
        
        # Resumen completo
        with open(f"{OUTPUT_DIR}/tablas/modelo_{nombre_archivo}.txt", 
                  'w', encoding='utf-8') as f:
            f.write(modelo.summary.as_text())
        
        # Coeficientes en Excel
        df_coefs = pd.DataFrame({
            'Variable': modelo.params.index,
            'Coeficiente': modelo.params.values,
            'SE': modelo.std_errors.values,
            't-stat': modelo.tstats.values,
            'p-valor': modelo.pvalues.values,
            'Sig': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else '' 
                    for p in modelo.pvalues.values]
        })
        df_coefs.to_excel(
            f"{OUTPUT_DIR}/tablas/coeficientes_{nombre_archivo}.xlsx",
            index=False
        )
        
        print(f"\n💾 Resultados guardados en: {OUTPUT_DIR}/tablas/")
    
    return resultados

# ==========================================
# FUNCIÓN: ESTIMAR PANEL DESAGREGADO
# ==========================================

def estimar_panel_por_grupo(df, y_col, x_cols, entity_col, time_col,
                             grupo_col, nombre_panel="Panel"):
    """
    Estima modelos de panel separados por cada valor de grupo_col.
    Útil para Panel 2 desagregado por TipodeFondo.
    
    Parameters:
        df (DataFrame): DataFrame completo
        y_col (str): Variable dependiente
        x_cols (list): Variables independientes
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        grupo_col (str): Columna de agrupación (ej. TipodeFondo)
        nombre_panel (str): Nombre base para archivos
    
    Returns:
        dict: Diccionario con resultados por grupo
    """
    print(f"\n{'='*70}")
    print(f"ESTIMACIÓN DESAGREGADA - {nombre_panel}")
    print(f"Desagregado por: {grupo_col}")
    print(f"{'='*70}")
    
    grupos = sorted(df[grupo_col].unique())
    resultados_por_grupo = {}
    
    for grupo in grupos:
        print(f"\n{'─'*70}")
        print(f"📌 Estimando para {grupo_col} = {grupo}")
        print(f"{'─'*70}")
        
        # Filtrar datos
        df_grupo = df[df[grupo_col] == grupo].copy()
        
        # Estimar
        resultado = estimar_panel_restringido(
            df_grupo,
            y_col,
            x_cols,
            entity_col,
            time_col,
            nombre_panel=f"{nombre_panel}_{grupo_col}_{grupo}",
            guardar=True
        )
        
        resultados_por_grupo[grupo] = resultado
    
    return resultados_por_grupo

# ==========================================
# FUNCIÓN: CREAR TABLA COMPARATIVA
# ==========================================

def crear_tabla_comparativa(resultados_dict, nombre_archivo="tabla_comparativa"):
    """
    Crea tabla comparativa estilo Stargazer con múltiples modelos.
    
    Parameters:
        resultados_dict (dict): Diccionario {nombre_modelo: resultado}
        nombre_archivo (str): Nombre base del archivo
    """
    print(f"\n{'='*70}")
    print("CREANDO TABLA COMPARATIVA")
    print(f"{'='*70}")
    
    # Extraer coeficientes de cada modelo
    df_comparativa = pd.DataFrame()
    
    for nombre_modelo, resultado in resultados_dict.items():
        df_temp = pd.DataFrame({
            f'{nombre_modelo}_coef': resultado['coefs'],
            f'{nombre_modelo}_se': resultado['se'],
            f'{nombre_modelo}_pval': resultado['pvals']
        })
        
        if df_comparativa.empty:
            df_comparativa = df_temp
        else:
            df_comparativa = df_comparativa.join(df_temp, how='outer')
    
    # Guardar
    df_comparativa.to_excel(f"{OUTPUT_DIR}/tablas/{nombre_archivo}.xlsx")
    print(f"💾 Tabla comparativa guardada: {OUTPUT_DIR}/tablas/{nombre_archivo}.xlsx")
    
    return df_comparativa

# ==========================================
# EJECUTAR ESTIMACIONES
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FASE 3.2: ESTIMACIÓN DE MODELOS FINALES")
    print("="*70)
    
    # ==========================================
    # 1. PANEL 1: APORTES (MODELO RESTRINGIDO)
    # ==========================================
    
    print("\n🎯 PASO 1: Estimación Panel 1 (Aportes AFP)")
    print("\nSegún diagnóstico previo (Test F p=0.816):")
    print("→ Modelo RESTRINGIDO (sin interacciones de control con COVID)")
    
    # Variables X (modelo restringido: SIN interacciones de control)
    x_vars_p1_final = CONFIG.VARS_IV_MOD + ctrl_c_p1 + mes_p1
    
    # Estimar
    modelo_p1 = estimar_panel_restringido(
        df_p1,
        CONFIG.VAR_Y_PANEL1,
        x_vars_p1_final,
        CONFIG.COL_AFP,
        CONFIG.COL_FECHA,
        nombre_panel="PANEL_1_APORTES",
        guardar=True
    )
    
    print(f"\n✅ Panel 1 estimado:")
    print(f"   • R² Within: {modelo_p1['r2_within']:.4f}")
    print(f"   • F-stat: {modelo_p1['f_stat']:.4f} (p={modelo_p1['f_pval']:.4f})")
    
    # ==========================================
    # 2. PANEL 2: REASIGNACIÓN (MODELO RESTRINGIDO AGREGADO)
    # ==========================================
    
    print("\n\n🎯 PASO 2: Estimación Panel 2 (Reasignación) - AGREGADO")
    print("\nSegún diagnóstico previo (Test Wald p=0.9958):")
    print("→ Modelo RESTRINGIDO (sin interacciones de control con COVID)")
    
    x_vars_p2_final = CONFIG.VARS_IV_MOD + ctrl_c_p2 + mes_p2
    
    modelo_p2_agregado = estimar_panel_restringido(
        df_p2,
        CONFIG.VAR_Y_PANEL2,
        x_vars_p2_final,
        'Entidad_Compuesta',
        CONFIG.COL_FECHA,
        nombre_panel="PANEL_2_REASIGNACION_AGREGADO",
        guardar=True
    )
    
    print(f"\n✅ Panel 2 Agregado estimado:")
    print(f"   • R² Within: {modelo_p2_agregado['r2_within']:.4f}")
    print(f"   • F-stat: {modelo_p2_agregado['f_stat']:.4f} (p={modelo_p2_agregado['f_pval']:.4f})")
    
    # ==========================================
    # 3. PANEL 2: DESAGREGADO POR TIPO DE FONDO
    # ==========================================
    
    print("\n\n🎯 PASO 3: Estimación Panel 2 - DESAGREGADO POR FONDO")
    print("\nPara testear H2 (Flight-to-Quality):")
    print("→ Comparar β₁(PC1_Global) entre Fondo 0 (conservador) y Fondo 3 (agresivo)")
    
    modelos_p2_por_fondo = estimar_panel_por_grupo(
        df_p2,
        CONFIG.VAR_Y_PANEL2,
        x_vars_p2_final,
        'Entidad_Compuesta',
        CONFIG.COL_FECHA,
        CONFIG.COL_FONDO,
        nombre_panel="PANEL_2_REASIGNACION"
    )
    
    print(f"\n✅ Panel 2 Desagregado estimado:")
    for fondo, resultado in modelos_p2_por_fondo.items():
        coef_global = resultado['coefs'].get('PC1_Global_c', np.nan)
        pval_global = resultado['pvals'].get('PC1_Global_c', np.nan)
        print(f"   • Fondo {fondo}: β₁(Global) = {coef_global:.4f} (p={pval_global:.4f})")
    
    # ==========================================
    # 4. CREAR TABLA COMPARATIVA PANEL 2
    # ==========================================
    
    print("\n\n🎯 PASO 4: Creando tabla comparativa Panel 2")
    
    # Combinar agregado + por fondo
    resultados_p2_todos = {
        'Agregado': modelo_p2_agregado,
        **{f'Fondo_{k}': v for k, v in modelos_p2_por_fondo.items()}
    }
    
    tabla_comparativa_p2 = crear_tabla_comparativa(
        resultados_p2_todos,
        nombre_archivo="tabla_comparativa_panel2"
    )
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    
    print("\n" + "="*70)
    print("✅ FASE 3.2 COMPLETADA")
    print("="*70)
    print("\n📊 MODELOS ESTIMADOS:")
    print(f"\n✅ Panel 1 (Aportes):")
    print(f"   • Modelo: RESTRINGIDO")
    print(f"   • N obs: {modelo_p1['n_obs']}")
    print(f"   • R² Within: {modelo_p1['r2_within']:.4f}")
    print(f"\n✅ Panel 2 (Reasignación):")
    print(f"   • Modelo Agregado: RESTRINGIDO")
    print(f"   • N obs: {modelo_p2_agregado['n_obs']}")
    print(f"   • R² Within: {modelo_p2_agregado['r2_within']:.4f}")
    print(f"   • Modelos por Fondo: {len(modelos_p2_por_fondo)}")
    print("\n📁 Archivos generados:")
    print(f"   • {OUTPUT_DIR}/tablas/modelo_panel_1_aportes.txt")
    print(f"   • {OUTPUT_DIR}/tablas/coeficientes_panel_1_aportes.xlsx")
    print(f"   • {OUTPUT_DIR}/tablas/modelo_panel_2_reasignacion_agregado.txt")
    print(f"   • {OUTPUT_DIR}/tablas/tabla_comparativa_panel2.xlsx")
    print("\n📌 Próximo paso: Ejecutar modulo_fase33_supuestos.py")
    print("="*70)
    
    # Exportar variables para siguiente módulo
    resultados_modelos = {
        'panel1': modelo_p1,
        'panel2_agregado': modelo_p2_agregado,
        'panel2_por_fondo': modelos_p2_por_fondo
    }
