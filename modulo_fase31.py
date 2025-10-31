"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO FASE 3.1: VERIFICACIÓN PRE-ESTIMACIÓN
═══════════════════════════════════════════════════════════════════════════════
Objetivos (Kerlinger 1986):
1. Verificar varianza suficiente en todas las variables
2. Validar entidades sin varianza nula en Panel 2
3. Test de Hausman para justificar Efectos Fijos (FE vs RE)
═══════════════════════════════════════════════════════════════════════════════
"""

# Importar configuración del Módulo 0
try:
    from modulo_0_config import (CONFIG, OUTPUT_DIR, datos_raw, 
                                  df_p1, df_p2, ctrl_c_p1, ctrl_c_p2,
                                  mes_p1, mes_p2)
    print("✅ Configuración importada desde Módulo 0")
except ImportError:
    print("⚠️  Ejecuta primero: modulo_0_config.py")
    raise

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects, compare

# ==========================================
# FUNCIÓN: VERIFICAR VARIANZA
# ==========================================

def verificar_varianza(df, columnas, nombre_panel="Panel"):
    """
    Verifica varianza suficiente en variables según Kerlinger (1986).
    
    Criterios de Validación:
    - Varianza mínima > 0.001
    - NAs máximos < 5% de observaciones
    
    Parameters:
        df (DataFrame): DataFrame con los datos
        columnas (list): Lista de columnas a verificar
        nombre_panel (str): Nombre del panel para reporte
    
    Returns:
        dict: Resultados de verificación con alertas
    """
    print(f"\n{'='*70}")
    print(f"VERIFICACIÓN DE VARIANZA - {nombre_panel}")
    print(f"{'='*70}")
    
    resultados = []
    alertas = []
    
    for col in columnas:
        if col not in df.columns:
            print(f"⚠️  ADVERTENCIA: '{col}' no encontrada en {nombre_panel}")
            continue
            
        n_total = len(df)
        n_na = df[col].isna().sum()
        pct_na = (n_na / n_total) * 100
        var = df[col].var()
        
        # Percentiles
        p5, p95 = df[col].quantile([0.05, 0.95])
        
        # Estado
        estado = "✅ OK"
        if pct_na > 5:
            estado = "⚠️  ALTA FALTANCIA"
            alertas.append(f"{col}: {pct_na:.1f}% NAs")
        if var < 0.001:
            estado = "❌ VARIANZA NULA"
            alertas.append(f"{col}: Varianza ≈ 0")
        
        resultados.append({
            'Variable': col,
            'N': n_total - n_na,
            'NAs': n_na,
            '% NAs': pct_na,
            'Varianza': var,
            'P5': p5,
            'P95': p95,
            'Estado': estado
        })
    
    df_resultado = pd.DataFrame(resultados)
    print(df_resultado.to_string(index=False))
    
    # Resumen
    print(f"\n{'='*70}")
    if len(alertas) == 0:
        print(f"✅ TODAS LAS VARIABLES PASAN LA VERIFICACIÓN")
    else:
        print(f"⚠️  {len(alertas)} ALERTAS DETECTADAS:")
        for alerta in alertas:
            print(f"   • {alerta}")
    print(f"{'='*70}")
    
    return {
        'resultados': df_resultado,
        'alertas': alertas,
        'n_variables': len(columnas),
        'n_alertas': len(alertas)
    }

# ==========================================
# FUNCIÓN: VALIDAR ENTIDADES PANEL 2
# ==========================================

def validar_entidades_panel2(df, col_entidad, var_y):
    """
    Valida que ninguna entidad tenga varianza nula en la VD (Panel 2).
    
    Crítico para Efectos Fijos: Entidades sin varianza causan colinealidad perfecta.
    
    Parameters:
        df (DataFrame): DataFrame del Panel 2
        col_entidad (str): Nombre de columna de entidad
        var_y (str): Variable dependiente
    
    Returns:
        dict: Resultados con entidades problemáticas
    """
    print(f"\n{'='*70}")
    print(f"VALIDACIÓN DE ENTIDADES - PANEL 2")
    print(f"{'='*70}")
    
    # Calcular varianza por entidad
    varianzas = df.groupby(col_entidad)[var_y].var().sort_values()
    
    print(f"\n📊 Varianza de '{var_y}' por entidad:\n")
    print(varianzas.to_string())
    
    # Identificar entidades problemáticas
    entidades_nulas = varianzas[varianzas < 0.001].index.tolist()
    
    print(f"\n{'='*70}")
    if len(entidades_nulas) == 0:
        print(f"✅ TODAS LAS ENTIDADES TIENEN VARIANZA SUFICIENTE")
    else:
        print(f"⚠️  {len(entidades_nulas)} ENTIDADES CON VARIANZA NULA:")
        for ent in entidades_nulas:
            print(f"   • {ent}: var = {varianzas[ent]:.6f}")
        print(f"\n💡 RECOMENDACIÓN: Excluir estas entidades del análisis FE")
    print(f"{'='*70}")
    
    return {
        'varianzas': varianzas,
        'entidades_nulas': entidades_nulas,
        'n_entidades_ok': len(varianzas) - len(entidades_nulas),
        'n_entidades_problema': len(entidades_nulas)
    }

# ==========================================
# FUNCIÓN: TEST DE HAUSMAN
# ==========================================

def test_hausman_panel(df, y_col, x_cols, entity_col, time_col, nombre_panel="Panel"):
    """
    Ejecuta el Test de Hausman para validar Efectos Fijos vs Efectos Aleatorios.
    
    H0: Efectos Aleatorios son consistentes (diferencia no sistemática)
    HA: Efectos Fijos son necesarios (diferencia sistemática)
    
    ESPERADO: Rechazar H0 (p < 0.05) → Usar FE
    
    Justificación teórica:
    Si las características no observadas de las entidades están correlacionadas
    con los regresores, RE es inconsistente y FE es necesario.
    
    Parameters:
        df (DataFrame): DataFrame con MultiIndex (entity, time)
        y_col (str): Variable dependiente
        x_cols (list): Variables independientes
        entity_col (str): Nombre de entidad (para referencia)
        time_col (str): Nombre de tiempo (para referencia)
        nombre_panel (str): Nombre para reportes
    
    Returns:
        dict: Resultados del test con modelos FE y RE
    """
    print(f"\n{'='*70}")
    print(f"TEST DE HAUSMAN - {nombre_panel}")
    print(f"{'='*70}")
    print("Efectos Fijos (FE) vs Efectos Aleatorios (RE)")
    
    try:
        # Preparar variables
        y = df[y_col]
        X = df[x_cols]
        
        print(f"\n📊 Dimensiones:")
        print(f"   • Observaciones: {len(y)}")
        print(f"   • Entidades: {y.index.get_level_values(0).nunique()}")
        print(f"   • Períodos: {y.index.get_level_values(1).nunique()}")
        
        # Modelo con Efectos Fijos
        print("\n🔄 Estimando modelo con Efectos Fijos (FE)...")
        mod_fe = PanelOLS(y, X, entity_effects=True).fit(cov_type='robust')
        
        # Modelo con Efectos Aleatorios
        print("🔄 Estimando modelo con Efectos Aleatorios (RE)...")
        mod_re = RandomEffects(y, X).fit(cov_type='robust')
        
        # Comparación de Hausman
        print("\n🔬 Ejecutando Test de Hausman...")
        comparacion = compare({'FE': mod_fe, 'RE': mod_re})
        
        print(f"\n{'='*70}")
        print("RESULTADOS DEL TEST DE HAUSMAN")
        print(f"{'='*70}")
        print(comparacion.summary)
        
        # Interpretación
        print(f"\n{'='*70}")
        print("INTERPRETACIÓN")
        print(f"{'='*70}")
        print(f"\nR² Within (varianza explicada dentro de entidades):")
        print(f"  • Efectos Fijos (FE):      {mod_fe.rsquared_within:.4f}")
        print(f"  • Efectos Aleatorios (RE): {mod_re.rsquared_within:.4f}")
        
        print(f"\n💡 CRITERIO DE DECISIÓN:")
        print(f"   Si Test de Hausman rechaza H0 (p < 0.05):")
        print(f"   → Usar EFECTOS FIJOS (entity_effects=True)")
        print(f"\n   Justificación: Las características no observadas de las")
        print(f"   entidades están correlacionadas con los regresores.")
        print(f"   RE sería inconsistente bajo esta correlación.")
        print(f"{'='*70}")
        
        return {
            'mod_fe': mod_fe,
            'mod_re': mod_re,
            'comparacion': comparacion,
            'r2_fe': mod_fe.rsquared_within,
            'r2_re': mod_re.rsquared_within,
            'nombre_panel': nombre_panel
        }
        
    except Exception as e:
        print(f"\n❌ ERROR en Test de Hausman: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# EJECUTAR VERIFICACIONES
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FASE 3.1: VERIFICACIÓN PRE-ESTIMACIÓN")
    print("="*70)
    
    # ==========================================
    # 1. VERIFICACIÓN DE VARIANZA - PANEL 1
    # ==========================================
    
    vars_verificar_p1 = CONFIG.VARS_IV_MOD + ctrl_c_p1 + mes_p1
    
    print("\n🔍 PASO 1: Verificación de varianza Panel 1...")
    resultado_var_p1 = verificar_varianza(df_p1, vars_verificar_p1, "PANEL 1: APORTES")
    
    # Guardar resultados
    resultado_var_p1['resultados'].to_excel(
        f"{OUTPUT_DIR}/diagnosticos/verificacion_varianza_panel1.xlsx", 
        index=False
    )
    print(f"💾 Guardado: {OUTPUT_DIR}/diagnosticos/verificacion_varianza_panel1.xlsx")
    
    # ==========================================
    # 2. VERIFICACIÓN DE VARIANZA - PANEL 2
    # ==========================================
    
    vars_verificar_p2 = CONFIG.VARS_IV_MOD + ctrl_c_p2 + mes_p2
    
    print("\n🔍 PASO 2: Verificación de varianza Panel 2...")
    resultado_var_p2 = verificar_varianza(df_p2, vars_verificar_p2, "PANEL 2: REASIGNACIÓN")
    
    # Validar entidades
    print("\n🔍 PASO 3: Validación de entidades Panel 2...")
    resultado_ent_p2 = validar_entidades_panel2(
        df_p2, 
        'Entidad_Compuesta', 
        CONFIG.VAR_Y_PANEL2
    )
    
    # Guardar resultados
    resultado_var_p2['resultados'].to_excel(
        f"{OUTPUT_DIR}/diagnosticos/verificacion_varianza_panel2.xlsx",
        index=False
    )
    resultado_ent_p2['varianzas'].to_excel(
        f"{OUTPUT_DIR}/diagnosticos/validacion_entidades_panel2.xlsx"
    )
    print(f"💾 Guardado: {OUTPUT_DIR}/diagnosticos/verificacion_varianza_panel2.xlsx")
    print(f"💾 Guardado: {OUTPUT_DIR}/diagnosticos/validacion_entidades_panel2.xlsx")
    
    # ==========================================
    # 3. TEST DE HAUSMAN - PANEL 1
    # ==========================================
    
    print("\n🔬 PASO 4: Test de Hausman Panel 1...")
    
    # Preparar datos con MultiIndex
    df_p1_idx = df_p1.dropna(subset=[CONFIG.VAR_Y_PANEL1] + vars_verificar_p1)
    df_p1_idx = df_p1_idx.set_index([CONFIG.COL_AFP, CONFIG.COL_FECHA])
    
    # Variables X (modelo restringido)
    x_vars_p1 = CONFIG.VARS_IV_MOD + ctrl_c_p1 + mes_p1
    
    # Ejecutar Hausman
    resultado_hausman_p1 = test_hausman_panel(
        df_p1_idx,
        CONFIG.VAR_Y_PANEL1,
        x_vars_p1,
        CONFIG.COL_AFP,
        CONFIG.COL_FECHA,
        "PANEL 1: APORTES"
    )
    
    # Guardar resumen
    if resultado_hausman_p1:
        with open(f"{OUTPUT_DIR}/diagnosticos/hausman_panel1.txt", 'w', encoding='utf-8') as f:
            f.write("TEST DE HAUSMAN - PANEL 1: APORTES\n")
            f.write("="*70 + "\n\n")
            f.write("H0: Efectos Aleatorios son consistentes\n")
            f.write("HA: Efectos Fijos son necesarios\n\n")
            f.write(str(resultado_hausman_p1['comparacion'].summary))
            f.write("\n\n" + "="*70 + "\n")
            f.write(f"R² Within FE:  {resultado_hausman_p1['r2_fe']:.4f}\n")
            f.write(f"R² Within RE:  {resultado_hausman_p1['r2_re']:.4f}\n")
        print(f"💾 Guardado: {OUTPUT_DIR}/diagnosticos/hausman_panel1.txt")
    
    # ==========================================
    # 4. TEST DE HAUSMAN - PANEL 2
    # ==========================================
    
    print("\n🔬 PASO 5: Test de Hausman Panel 2...")
    
    df_p2_idx = df_p2.dropna(subset=[CONFIG.VAR_Y_PANEL2] + vars_verificar_p2)
    df_p2_idx = df_p2_idx.set_index(['Entidad_Compuesta', CONFIG.COL_FECHA])
    
    x_vars_p2 = CONFIG.VARS_IV_MOD + ctrl_c_p2 + mes_p2
    
    resultado_hausman_p2 = test_hausman_panel(
        df_p2_idx,
        CONFIG.VAR_Y_PANEL2,
        x_vars_p2,
        'Entidad_Compuesta',
        CONFIG.COL_FECHA,
        "PANEL 2: REASIGNACIÓN"
    )
    
    if resultado_hausman_p2:
        with open(f"{OUTPUT_DIR}/diagnosticos/hausman_panel2.txt", 'w', encoding='utf-8') as f:
            f.write("TEST DE HAUSMAN - PANEL 2: REASIGNACIÓN\n")
            f.write("="*70 + "\n\n")
            f.write("H0: Efectos Aleatorios son consistentes\n")
            f.write("HA: Efectos Fijos son necesarios\n\n")
            f.write(str(resultado_hausman_p2['comparacion'].summary))
            f.write("\n\n" + "="*70 + "\n")
            f.write(f"R² Within FE:  {resultado_hausman_p2['r2_fe']:.4f}\n")
            f.write(f"R² Within RE:  {resultado_hausman_p2['r2_re']:.4f}\n")
        print(f"💾 Guardado: {OUTPUT_DIR}/diagnosticos/hausman_panel2.txt")
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    
    print("\n" + "="*70)
    print("✅ FASE 3.1 COMPLETADA")
    print("="*70)
    print("\n📊 RESUMEN DE VERIFICACIONES:")
    print(f"\n✅ Panel 1:")
    print(f"   • Variables verificadas: {resultado_var_p1['n_variables']}")
    print(f"   • Alertas: {resultado_var_p1['n_alertas']}")
    print(f"\n✅ Panel 2:")
    print(f"   • Variables verificadas: {resultado_var_p2['n_variables']}")
    print(f"   • Alertas: {resultado_var_p2['n_alertas']}")
    print(f"   • Entidades problemáticas: {resultado_ent_p2['n_entidades_problema']}")
    print(f"\n✅ Test de Hausman:")
    print(f"   • Panel 1 completado: {'✓' if resultado_hausman_p1 else '✗'}")
    print(f"   • Panel 2 completado: {'✓' if resultado_hausman_p2 else '✗'}")
    print("\n📌 Próximo paso: Ejecutar modulo_fase32_estimacion.py")
    print("="*70)
