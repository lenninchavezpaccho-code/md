"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO FASE 3.4: ANÁLISIS DE ROBUSTEZ
═══════════════════════════════════════════════════════════════════════════════
5 Análisis de Robustez según validación doctoral:

1. Sin período COVID (datos hasta 2020-02)
2. Solo PC1_Global y Solo PC1_Sistemático (por separado)
3. Ventanas COVID alternativas (2020-02, 2020-04, 2020-06)
4. Sin trimestres extremos (excluir Q1 2020)
5. Modelo dinámico (incluir VD rezagada - solo Panel 1)

Objetivo: Verificar que los resultados principales sean ROBUSTOS a
          especificaciones alternativas.
═══════════════════════════════════════════════════════════════════════════════
"""

# Importar configuración
try:
    from modulo_0_config import (CONFIG, OUTPUT_DIR, datos_raw, 
                                  preparar_panel_para_analisis)
    from modulo_fase32_estimacion import estimar_panel_restringido
    print("✅ Configuración importada")
except ImportError:
    print("⚠️  Ejecuta primero: modulo_0_config.py y modulo_fase32_estimacion.py")
    raise

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# ==========================================
# ROBUSTEZ 1: SIN PERÍODO COVID
# ==========================================

def robustez_sin_covid(df_panel, df_pred, df_ctrl, var_y, entity_col, 
                       time_col, fecha_corte="2020-02-01", nombre_panel="Panel"):
    """
    Estima modelo excluyendo período COVID.
    
    Objetivo: Verificar que efectos pre-COVID sean consistentes con modelo completo.
    
    Parameters:
        df_panel (DataFrame): Panel principal
        df_pred (DataFrame): Predictores
        df_ctrl (DataFrame): Controles
        var_y (str): Variable dependiente
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        fecha_corte (str): Fecha hasta la cual incluir datos
        nombre_panel (str): Nombre del panel
    
    Returns:
        dict: Resultados del modelo sin COVID
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTEZ 1: SIN PERÍODO COVID - {nombre_panel}")
    print(f"{'='*70}")
    print(f"Fecha de corte: {fecha_corte}")
    
    # Filtrar datos hasta fecha de corte
    df_panel_filtrado = df_panel[df_panel[time_col] < fecha_corte].copy()
    df_pred_filtrado = df_pred[df_pred[time_col] < fecha_corte].copy()
    df_ctrl_filtrado = df_ctrl[df_ctrl[time_col] < fecha_corte].copy()
    
    print(f"\n📊 Observaciones:")
    print(f"   • Original: {len(df_panel)}")
    print(f"   • Sin COVID: {len(df_panel_filtrado)}")
    print(f"   • Reducción: {len(df_panel) - len(df_panel_filtrado)} obs")
    
    # Preparar datos
    df_prep, ctrl_c, ctrl_int, mes_dummies = preparar_panel_para_analisis(
        df_panel_filtrado, df_pred_filtrado, df_ctrl_filtrado, time_col
    )
    
    # Variables X (sin interacciones COVID ya que no hay período COVID)
    # Pero mantenemos las mismas variables para comparabilidad
    x_vars = CONFIG.VARS_IV_MOD + ctrl_c + mes_dummies
    
    # Estimar modelo
    resultado = estimar_panel_restringido(
        df_prep, var_y, x_vars, entity_col, time_col,
        nombre_panel=f"{nombre_panel}_sin_COVID",
        guardar=True
    )
    
    return resultado

# ==========================================
# ROBUSTEZ 2: SOLO PC1_GLOBAL O PC1_SISTEMÁTICO
# ==========================================

def robustez_predictores_individuales(df, var_y, ctrl_c, mes_dummies, 
                                       entity_col, time_col, nombre_panel="Panel"):
    """
    Estima 2 modelos: uno solo con PC1_Global, otro solo con PC1_Sistemático.
    
    Objetivo: Aislar el efecto de cada dimensión de volatilidad.
    
    Parameters:
        df (DataFrame): DataFrame preparado
        var_y (str): Variable dependiente
        ctrl_c (list): Controles centrados
        mes_dummies (list): Dummies de mes
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        nombre_panel (str): Nombre del panel
    
    Returns:
        dict: Resultados de ambos modelos
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTEZ 2: PREDICTORES INDIVIDUALES - {nombre_panel}")
    print(f"{'='*70}")
    
    resultados = {}
    
    # Modelo 1: Solo PC1_Global
    print("\n🔹 Modelo A: Solo PC1_Global_c")
    x_vars_global = ['PC1_Global_c', 'D_COVID', 'Int_Global_COVID'] + ctrl_c + mes_dummies
    
    resultado_global = estimar_panel_restringido(
        df, var_y, x_vars_global, entity_col, time_col,
        nombre_panel=f"{nombre_panel}_solo_Global",
        guardar=True
    )
    resultados['solo_global'] = resultado_global
    
    # Modelo 2: Solo PC1_Sistemático
    print("\n🔹 Modelo B: Solo PC1_Sistematico_c")
    x_vars_sist = ['PC1_Sistematico_c', 'D_COVID', 'Int_Sistematico_COVID'] + ctrl_c + mes_dummies
    
    resultado_sist = estimar_panel_restringido(
        df, var_y, x_vars_sist, entity_col, time_col,
        nombre_panel=f"{nombre_panel}_solo_Sistematico",
        guardar=True
    )
    resultados['solo_sistematico'] = resultado_sist
    
    return resultados

# ==========================================
# ROBUSTEZ 3: VENTANAS COVID ALTERNATIVAS
# ==========================================

def robustez_ventanas_covid(df_panel, df_pred, df_ctrl, var_y, entity_col, 
                             time_col, fechas_inicio_covid, nombre_panel="Panel"):
    """
    Estima modelos con diferentes fechas de inicio de COVID.
    
    Objetivo: Verificar sensibilidad a la definición temporal de COVID.
    
    Parameters:
        df_panel (DataFrame): Panel principal
        df_pred (DataFrame): Predictores
        df_ctrl (DataFrame): Controles
        var_y (str): Variable dependiente
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        fechas_inicio_covid (list): Lista de fechas alternativas
        nombre_panel (str): Nombre del panel
    
    Returns:
        dict: Resultados por cada ventana
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTEZ 3: VENTANAS COVID ALTERNATIVAS - {nombre_panel}")
    print(f"{'='*70}")
    
    resultados = {}
    
    for fecha_inicio in fechas_inicio_covid:
        print(f"\n🔹 Ventana COVID: inicio {fecha_inicio}")
        
        # Crear nueva dummy COVID
        df_pred_temp = df_pred.copy()
        df_pred_temp['D_COVID'] = (df_pred_temp[time_col] >= fecha_inicio).astype(int)
        
        # Recrear interacciones
        df_pred_temp['Int_Global_COVID'] = df_pred_temp['PC1_Global_c'] * df_pred_temp['D_COVID']
        df_pred_temp['Int_Sistematico_COVID'] = df_pred_temp['PC1_Sistematico_c'] * df_pred_temp['D_COVID']
        
        # Preparar datos
        df_prep, ctrl_c, ctrl_int, mes_dummies = preparar_panel_para_analisis(
            df_panel, df_pred_temp, df_ctrl, time_col
        )
        
        # Variables X
        x_vars = CONFIG.VARS_IV_MOD + ctrl_c + mes_dummies
        
        # Estimar
        resultado = estimar_panel_restringido(
            df_prep, var_y, x_vars, entity_col, time_col,
            nombre_panel=f"{nombre_panel}_COVID_{fecha_inicio}",
            guardar=True
        )
        
        resultados[fecha_inicio] = resultado
    
    return resultados

# ==========================================
# ROBUSTEZ 4: SIN TRIMESTRES EXTREMOS
# ==========================================

def robustez_sin_extremos(df_panel, df_pred, df_ctrl, var_y, entity_col, 
                          time_col, fechas_excluir, nombre_panel="Panel"):
    """
    Estima modelo excluyendo trimestres con eventos extremos.
    
    Objetivo: Verificar que resultados no son artefactos de outliers temporales.
    
    Parameters:
        df_panel (DataFrame): Panel principal
        df_pred (DataFrame): Predictores
        df_ctrl (DataFrame): Controles
        var_y (str): Variable dependiente
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        fechas_excluir (tuple): (fecha_inicio_excluir, fecha_fin_excluir)
        nombre_panel (str): Nombre del panel
    
    Returns:
        dict: Resultados sin extremos
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTEZ 4: SIN TRIMESTRES EXTREMOS - {nombre_panel}")
    print(f"{'='*70}")
    print(f"Excluyendo período: {fechas_excluir[0]} a {fechas_excluir[1]}")
    
    # Filtrar datos excluyendo período
    fecha_ini_excl, fecha_fin_excl = fechas_excluir
    
    mask_panel = (df_panel[time_col] < fecha_ini_excl) | (df_panel[time_col] > fecha_fin_excl)
    mask_pred = (df_pred[time_col] < fecha_ini_excl) | (df_pred[time_col] > fecha_fin_excl)
    mask_ctrl = (df_ctrl[time_col] < fecha_ini_excl) | (df_ctrl[time_col] > fecha_fin_excl)
    
    df_panel_filtrado = df_panel[mask_panel].copy()
    df_pred_filtrado = df_pred[mask_pred].copy()
    df_ctrl_filtrado = df_ctrl[mask_ctrl].copy()
    
    print(f"\n📊 Observaciones:")
    print(f"   • Original: {len(df_panel)}")
    print(f"   • Sin extremos: {len(df_panel_filtrado)}")
    print(f"   • Excluidas: {len(df_panel) - len(df_panel_filtrado)} obs")
    
    # Preparar datos
    df_prep, ctrl_c, ctrl_int, mes_dummies = preparar_panel_para_analisis(
        df_panel_filtrado, df_pred_filtrado, df_ctrl_filtrado, time_col
    )
    
    # Variables X
    x_vars = CONFIG.VARS_IV_MOD + ctrl_c + mes_dummies
    
    # Estimar
    resultado = estimar_panel_restringido(
        df_prep, var_y, x_vars, entity_col, time_col,
        nombre_panel=f"{nombre_panel}_sin_extremos",
        guardar=True
    )
    
    return resultado

# ==========================================
# ROBUSTEZ 5: MODELO DINÁMICO (Solo Panel 1)
# ==========================================

def robustez_modelo_dinamico(df, var_y, x_cols, entity_col, time_col, nombre_panel="Panel"):
    """
    Estima modelo dinámico incluyendo VD rezagada.
    
    ADVERTENCIA: Modelo dinámico en panel puede tener Arellano-Bond bias.
    Este es un análisis exploratorio, no causal definitivo.
    
    Parameters:
        df (DataFrame): DataFrame preparado
        var_y (str): Variable dependiente
        x_cols (list): Variables independientes
        entity_col (str): Columna de entidad
        time_col (str): Columna de tiempo
        nombre_panel (str): Nombre del panel
    
    Returns:
        dict: Resultados del modelo dinámico
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTEZ 5: MODELO DINÁMICO - {nombre_panel}")
    print(f"{'='*70}")
    print(f"Especificación: {var_y}(t) ~ {var_y}(t-1) + X(t)")
    
    # Crear variable dependiente rezagada
    df_sorted = df.sort_values([entity_col, time_col])
    df_sorted[f'{var_y}_lag1'] = df_sorted.groupby(entity_col)[var_y].shift(1)
    
    # Eliminar NAs del rezago
    df_modelo = df_sorted.dropna(subset=[var_y, f'{var_y}_lag1'] + x_cols)
    
    print(f"\n📊 Observaciones:")
    print(f"   • Sin rezago: {len(df)}")
    print(f"   • Con rezago: {len(df_modelo)}")
    print(f"   • Pérdidas: {len(df) - len(df_modelo)} obs (primera obs por entidad)")
    
    # Preparar para PanelOLS
    df_modelo = df_modelo.set_index([entity_col, time_col])
    
    y = df_modelo[var_y]
    X = df_modelo[[f'{var_y}_lag1'] + x_cols]
    
    # Estimar modelo dinámico
    print(f"\n🔄 Estimando PanelOLS con VD rezagada...")
    modelo = PanelOLS(y, X, entity_effects=True).fit(cov_type='robust')
    
    print(f"\n{modelo.summary}")
    
    # Extraer coeficiente del rezago
    coef_lag = modelo.params[f'{var_y}_lag1']
    pval_lag = modelo.pvalues[f'{var_y}_lag1']
    
    print(f"\n💡 INTERPRETACIÓN COEFICIENTE REZAGADO:")
    print(f"   • β(lag): {coef_lag:.4f} (p={pval_lag:.4f})")
    
    if abs(coef_lag) < 0.5:
        print(f"   ✅ Persistencia baja/moderada (|β| < 0.5)")
        print(f"   → Efectos de X no son solo inercia de VD pasada")
    else:
        print(f"   ⚠️  Persistencia alta (|β| ≥ 0.5)")
        print(f"   → Gran parte de variación es inercia temporal")
    
    # Guardar
    with open(f"{OUTPUT_DIR}/robustez/modelo_dinamico_{nombre_panel.lower().replace(' ', '_')}.txt", 
              'w', encoding='utf-8') as f:
        f.write(modelo.summary.as_text())
    
    resultados = {
        'modelo': modelo,
        'n_obs': len(y),
        'r2_within': modelo.rsquared_within,
        'coef_lag': coef_lag,
        'pval_lag': pval_lag,
        'coefs': modelo.params,
        'pvals': modelo.pvalues
    }
    
    return resultados

# ==========================================
# FUNCIÓN: COMPARAR RESULTADOS DE ROBUSTEZ
# ==========================================

def comparar_robustez(resultados_dict, coef_interes, nombre_archivo="comparacion_robustez"):
    """
    Genera tabla comparativa de coeficientes clave entre modelos de robustez.
    
    Parameters:
        resultados_dict (dict): {nombre_modelo: resultado}
        coef_interes (list): Lista de coeficientes a comparar
        nombre_archivo (str): Nombre del archivo de salida
    
    Returns:
        DataFrame: Tabla comparativa
    """
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE ANÁLISIS DE ROBUSTEZ")
    print(f"{'='*70}")
    
    comparacion = []
    
    for nombre_modelo, resultado in resultados_dict.items():
        fila = {'Modelo': nombre_modelo}
        
        for coef in coef_interes:
            if coef in resultado['coefs'].index:
                val = resultado['coefs'][coef]
                pval = resultado['pvals'][coef]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                fila[f'{coef}_coef'] = f"{val:.4f}{sig}"
                fila[f'{coef}_pval'] = f"{pval:.4f}"
            else:
                fila[f'{coef}_coef'] = 'N/A'
                fila[f'{coef}_pval'] = 'N/A'
        
        fila['R2_within'] = f"{resultado['r2_within']:.4f}"
        fila['N_obs'] = resultado['n_obs']
        
        comparacion.append(fila)
    
    df_comparacion = pd.DataFrame(comparacion)
    
    # Guardar
    df_comparacion.to_excel(f"{OUTPUT_DIR}/robustez/{nombre_archivo}.xlsx", index=False)
    print(f"\n💾 Comparación guardada: robustez/{nombre_archivo}.xlsx")
    
    # Mostrar en consola
    print(f"\n📊 TABLA COMPARATIVA:\n")
    print(df_comparacion.to_string(index=False))
    
    return df_comparacion

# ==========================================
# EJECUTAR ANÁLISIS DE ROBUSTEZ
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FASE 3.4: ANÁLISIS DE ROBUSTEZ")
    print("="*70)
    
    # Preparar datos base
    from modulo_0_config import df_p1, df_p2, ctrl_c_p1, ctrl_c_p2, mes_p1, mes_p2
    
    resultados_robustez_p1 = {}
    resultados_robustez_p2 = {}
    
    # ==========================================
    # PANEL 1: APORTES - 5 ANÁLISIS
    # ==========================================
    
    print("\n🔍 ANÁLISIS DE ROBUSTEZ - PANEL 1: APORTES")
    print("="*70)
    
    # Robustez 1: Sin COVID
    print("\n📌 Robustez 1/5: Sin período COVID")
    rob1_p1 = robustez_sin_covid(
        datos_raw['panel1'], datos_raw['predictores'], datos_raw['controles'],
        CONFIG.VAR_Y_PANEL1, CONFIG.COL_AFP, CONFIG.COL_FECHA,
        fecha_corte="2020-03-01", nombre_panel="Panel 1"
    )
    resultados_robustez_p1['Sin_COVID'] = rob1_p1
    
    # Robustez 2: Predictores individuales
    print("\n📌 Robustez 2/5: Predictores individuales")
    rob2_p1 = robustez_predictores_individuales(
        df_p1, CONFIG.VAR_Y_PANEL1, ctrl_c_p1, mes_p1,
        CONFIG.COL_AFP, CONFIG.COL_FECHA, nombre_panel="Panel 1"
    )
    resultados_robustez_p1.update({
        'Solo_Global': rob2_p1['solo_global'],
        'Solo_Sistematico': rob2_p1['solo_sistematico']
    })
    
    # Robustez 3: Ventanas COVID
    print("\n📌 Robustez 3/5: Ventanas COVID alternativas")
    fechas_covid = ["2020-02-01", "2020-04-01", "2020-06-01"]
    rob3_p1 = robustez_ventanas_covid(
        datos_raw['panel1'], datos_raw['predictores'], datos_raw['controles'],
        CONFIG.VAR_Y_PANEL1, CONFIG.COL_AFP, CONFIG.COL_FECHA,
        fechas_covid, nombre_panel="Panel 1"
    )
    for fecha, resultado in rob3_p1.items():
        resultados_robustez_p1[f'COVID_{fecha}'] = resultado
    
    # Robustez 4: Sin extremos
    print("\n📌 Robustez 4/5: Sin trimestres extremos (Q1 2020)")
    rob4_p1 = robustez_sin_extremos(
        datos_raw['panel1'], datos_raw['predictores'], datos_raw['controles'],
        CONFIG.VAR_Y_PANEL1, CONFIG.COL_AFP, CONFIG.COL_FECHA,
        fechas_excluir=("2020-01-01", "2020-03-31"),
        nombre_panel="Panel 1"
    )
    resultados_robustez_p1['Sin_Q1_2020'] = rob4_p1
    
    # Robustez 5: Modelo dinámico
    print("\n📌 Robustez 5/5: Modelo dinámico")
    x_vars_p1 = CONFIG.VARS_IV_MOD + ctrl_c_p1 + mes_p1
    rob5_p1 = robustez_modelo_dinamico(
        df_p1, CONFIG.VAR_Y_PANEL1, x_vars_p1,
        CONFIG.COL_AFP, CONFIG.COL_FECHA, nombre_panel="Panel 1"
    )
    resultados_robustez_p1['Dinamico'] = rob5_p1
    
    # ==========================================
    # PANEL 2: REASIGNACIÓN - 4 ANÁLISIS (sin dinámico)
    # ==========================================
    
    print("\n\n🔍 ANÁLISIS DE ROBUSTEZ - PANEL 2: REASIGNACIÓN")
    print("="*70)
    print("Nota: Modelo dinámico omitido (VD en diferencias, menos interpretable)")
    
    # Robustez 1: Sin COVID
    print("\n📌 Robustez 1/4: Sin período COVID")
    rob1_p2 = robustez_sin_covid(
        datos_raw['panel2'], datos_raw['predictores'], datos_raw['controles'],
        CONFIG.VAR_Y_PANEL2, 'Entidad_Compuesta', CONFIG.COL_FECHA,
        fecha_corte="2020-03-01", nombre_panel="Panel 2"
    )
    resultados_robustez_p2['Sin_COVID'] = rob1_p2
    
    # Robustez 2: Predictores individuales
    print("\n📌 Robustez 2/4: Predictores individuales")
    rob2_p2 = robustez_predictores_individuales(
        df_p2, CONFIG.VAR_Y_PANEL2, ctrl_c_p2, mes_p2,
        'Entidad_Compuesta', CONFIG.COL_FECHA, nombre_panel="Panel 2"
    )
    resultados_robustez_p2.update({
        'Solo_Global': rob2_p2['solo_global'],
        'Solo_Sistematico': rob2_p2['solo_sistematico']
    })
    
    # Robustez 3: Ventanas COVID
    print("\n📌 Robustez 3/4: Ventanas COVID alternativas")
    rob3_p2 = robustez_ventanas_covid(
        datos_raw['panel2'], datos_raw['predictores'], datos_raw['controles'],
        CONFIG.VAR_Y_PANEL2, 'Entidad_Compuesta', CONFIG.COL_FECHA,
        fechas_covid, nombre_panel="Panel 2"
    )
    for fecha, resultado in rob3_p2.items():
        resultados_robustez_p2[f'COVID_{fecha}'] = resultado
    
    # Robustez 4: Sin extremos
    print("\n📌 Robustez 4/4: Sin trimestres extremos (Q1 2020)")
    rob4_p2 = robustez_sin_extremos(
        datos_raw['panel2'], datos_raw['predictores'], datos_raw['controles'],
        CONFIG.VAR_Y_PANEL2, 'Entidad_Compuesta', CONFIG.COL_FECHA,
        fechas_excluir=("2020-01-01", "2020-03-31"),
        nombre_panel="Panel 2"
    )
    resultados_robustez_p2['Sin_Q1_2020'] = rob4_p2
    
    # ==========================================
    # GENERAR COMPARACIONES
    # ==========================================
    
    print("\n\n📊 GENERANDO TABLAS COMPARATIVAS")
    
    # Coeficientes de interés
    coefs_interes = ['PC1_Global_c', 'PC1_Sistematico_c', 'Int_Global_COVID', 'Int_Sistematico_COVID']
    
    # Comparación Panel 1
    comp_p1 = comparar_robustez(
        resultados_robustez_p1,
        coefs_interes,
        nombre_archivo="comparacion_robustez_panel1"
    )
    
    # Comparación Panel 2
    comp_p2 = comparar_robustez(
        resultados_robustez_p2,
        coefs_interes,
        nombre_archivo="comparacion_robustez_panel2"
    )
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    
    print("\n" + "="*70)
    print("✅ FASE 3.4 COMPLETADA")
    print("="*70)
    print("\n📊 ANÁLISIS DE ROBUSTEZ COMPLETADOS:")
    print(f"   ✅ Panel 1: {len(resultados_robustez_p1)} especificaciones")
    print(f"   ✅ Panel 2: {len(resultados_robustez_p2)} especificaciones")
    
    print("\n📁 Archivos generados en robustez/:")
    print("   • comparacion_robustez_panel1.xlsx")
    print("   • comparacion_robustez_panel2.xlsx")
    print("   • modelo_*_sin_COVID.txt (×2)")
    print("   • modelo_*_solo_*.txt (×4)")
    print("   • modelo_*_COVID_*.txt (×6)")
    print("   • modelo_*_sin_extremos.txt (×2)")
    print("   • modelo_dinamico_panel_1.txt")
    
    print("\n💡 INTERPRETACIÓN GENERAL:")
    print("   Si coeficientes principales (PC1_Global, PC1_Sistemático)")
    print("   mantienen signo y significancia en mayoría de modelos:")
    print("   → Resultados son ROBUSTOS ✅")
    
    print("\n📌 Próximo paso: Ejecutar modulo_fase4_hipotesis.py")
    print("="*70)
