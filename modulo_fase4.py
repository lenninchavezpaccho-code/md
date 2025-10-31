"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO FASE 4: TEST DE HIPÓTESIS DE INVESTIGACIÓN
═══════════════════════════════════════════════════════════════════════════════
Operacionalización de hipótesis según Samaja (1994):

PANEL 1 (Aportes):
  H1.1: β₁(PC1_Global_c) < 0 y p < 0.05
  H1.2: β₂(PC1_Sistematico_c) ≠ 0 y p < 0.10
  H1.3: β₆(Int_Global_COVID) ≠ 0 y p < 0.10

PANEL 2 (Reasignación - Flight-to-Quality):
  H2.1: β₁(PC1_Global, Fondo0) > β₁(PC1_Global, Fondo3)
  H2.2: Todos los fondos: β₁ < 0 (signo negativo)

PANEL 3 (Composición - Presión Liquidez):
  H3.1: |β₁(Minería)| > |β₁(Soberano)|
  H3.2: |β₂(Local)| > |β₂(Extranjero)|
═══════════════════════════════════════════════════════════════════════════════
"""

# Importar configuración y resultados previos
try:
    from modulo_0_config import CONFIG, OUTPUT_DIR
    from modulo_fase32_estimacion import resultados_modelos
    print("✅ Configuración y modelos importados")
except ImportError:
    print("⚠️  Ejecuta primero: modulo_0_config.py y modulo_fase32_estimacion.py")
    raise

import pandas as pd
import numpy as np
from scipy import stats

# ==========================================
# FUNCIÓN: TEST DE HIPÓTESIS INDIVIDUAL
# ==========================================

def test_hipotesis(coef, se, tstat, pval, hipotesis_tipo, alpha=0.05, 
                    valor_comparacion=0, nombre_hipotesis="H"):
    """
    Testa una hipótesis individual sobre un coeficiente.
    
    Tipos de hipótesis:
    - 'menor': β < valor_comparacion (cola izquierda)
    - 'mayor': β > valor_comparacion (cola derecha)
    - 'distinto': β ≠ valor_comparacion (dos colas)
    
    Parameters:
        coef (float): Coeficiente estimado
        se (float): Error estándar
        tstat (float): Estadístico t
        pval (float): p-valor (dos colas)
        hipotesis_tipo (str): Tipo de hipótesis ('menor', 'mayor', 'distinto')
        alpha (float): Nivel de significancia
        valor_comparacion (float): Valor de H0 (usualmente 0)
        nombre_hipotesis (str): Etiqueta de la hipótesis
    
    Returns:
        dict: Resultado del test
    """
    # Ajustar p-valor según tipo de hipótesis
    if hipotesis_tipo == 'menor':
        # Cola izquierda: β < valor_comparacion
        pval_ajustado = pval / 2 if coef < valor_comparacion else 1 - (pval / 2)
        decision_texto = f"β < {valor_comparacion}"
    elif hipotesis_tipo == 'mayor':
        # Cola derecha: β > valor_comparacion
        pval_ajustado = pval / 2 if coef > valor_comparacion else 1 - (pval / 2)
        decision_texto = f"β > {valor_comparacion}"
    else:  # 'distinto'
        # Dos colas: β ≠ valor_comparacion
        pval_ajustado = pval
        decision_texto = f"β ≠ {valor_comparacion}"
    
    # Decisión
    rechaza_h0 = pval_ajustado < alpha
    
    # Significancia
    sig = '***' if pval_ajustado < 0.01 else '**' if pval_ajustado < 0.05 else '*' if pval_ajustado < 0.10 else ''
    
    resultado = {
        'hipotesis': nombre_hipotesis,
        'tipo': hipotesis_tipo,
        'coeficiente': coef,
        'se': se,
        'tstat': tstat,
        'pval': pval_ajustado,
        'alpha': alpha,
        'rechaza_h0': rechaza_h0,
        'decision': decision_texto,
        'sig': sig,
        'resultado': '✅ CONFIRMADA' if rechaza_h0 else '❌ NO CONFIRMADA'
    }
    
    return resultado

# ==========================================
# FUNCIÓN: TESTEAR HIPÓTESIS PANEL 1
# ==========================================

def testear_hipotesis_panel1(modelo, alpha=0.05):
    """
    Testa las 3 hipótesis del Panel 1 (Aportes).
    
    H1.1: β₁(PC1_Global_c) < 0 y p < 0.05
    H1.2: β₂(PC1_Sistematico_c) ≠ 0 y p < 0.10
    H1.3: β₆(Int_Global_COVID) ≠ 0 y p < 0.10
    
    Parameters:
        modelo (dict): Resultado del modelo Panel 1
        alpha (float): Nivel de significancia base
    
    Returns:
        dict: Resultados de los 3 tests
    """
    print(f"\n{'='*70}")
    print("TEST DE HIPÓTESIS - PANEL 1: APORTES")
    print(f"{'='*70}")
    
    resultados = {}
    
    # H1.1: Amplitud de volatilidad global reduce aportes
    print("\n📊 H1.1: Turbulencia global reduce aportes")
    print("   H0: β₁(PC1_Global_c) ≥ 0")
    print("   HA: β₁(PC1_Global_c) < 0 (p < 0.05)")
    
    coef_global = modelo['coefs']['PC1_Global_c']
    se_global = modelo['se']['PC1_Global_c']
    tstat_global = modelo['tstats']['PC1_Global_c']
    pval_global = modelo['pvals']['PC1_Global_c']
    
    h11 = test_hipotesis(
        coef_global, se_global, tstat_global, pval_global,
        'menor', alpha=0.05, valor_comparacion=0,
        nombre_hipotesis="H1.1"
    )
    resultados['H1.1'] = h11
    
    print(f"   • β₁ = {coef_global:.4f} (SE={se_global:.4f})")
    print(f"   • p-valor (cola izq): {h11['pval']:.4f}")
    print(f"   • {h11['resultado']}")
    
    # H1.2: Sensibilidad país amplifica/amortigua efecto
    print("\n📊 H1.2: Sensibilidad de Perú tiene efecto propio")
    print("   H0: β₂(PC1_Sistematico_c) = 0")
    print("   HA: β₂(PC1_Sistematico_c) ≠ 0 (p < 0.10)")
    
    coef_sist = modelo['coefs']['PC1_Sistematico_c']
    se_sist = modelo['se']['PC1_Sistematico_c']
    tstat_sist = modelo['tstats']['PC1_Sistematico_c']
    pval_sist = modelo['pvals']['PC1_Sistematico_c']
    
    h12 = test_hipotesis(
        coef_sist, se_sist, tstat_sist, pval_sist,
        'distinto', alpha=0.10, valor_comparacion=0,
        nombre_hipotesis="H1.2"
    )
    resultados['H1.2'] = h12
    
    print(f"   • β₂ = {coef_sist:.4f} (SE={se_sist:.4f})")
    print(f"   • p-valor (dos colas): {h12['pval']:.4f}")
    print(f"   • {h12['resultado']}")
    
    # H1.3: COVID moderó el efecto de volatilidad global
    print("\n📊 H1.3: COVID amplificó efecto de turbulencia global")
    print("   H0: β₆(Int_Global_COVID) = 0")
    print("   HA: β₆(Int_Global_COVID) ≠ 0 (p < 0.10)")
    
    coef_int = modelo['coefs']['Int_Global_COVID']
    se_int = modelo['se']['Int_Global_COVID']
    tstat_int = modelo['tstats']['Int_Global_COVID']
    pval_int = modelo['pvals']['Int_Global_COVID']
    
    h13 = test_hipotesis(
        coef_int, se_int, tstat_int, pval_int,
        'distinto', alpha=0.10, valor_comparacion=0,
        nombre_hipotesis="H1.3"
    )
    resultados['H1.3'] = h13
    
    print(f"   • β₆ = {coef_int:.4f} (SE={se_int:.4f})")
    print(f"   • p-valor (dos colas): {h13['pval']:.4f}")
    print(f"   • {h13['resultado']}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN PANEL 1:")
    confirmadas = sum([h['rechaza_h0'] for h in resultados.values()])
    print(f"   Hipótesis confirmadas: {confirmadas}/3")
    print(f"{'='*70}")
    
    return resultados

# ==========================================
# FUNCIÓN: TESTEAR HIPÓTESIS PANEL 2
# ==========================================

def testear_hipotesis_panel2(modelos_por_fondo, alpha=0.05):
    """
    Testa las 2 hipótesis del Panel 2 (Flight-to-Quality).
    
    H2.1: β₁(Fondo0) > β₁(Fondo3) - Fondos conservadores más resilientes
    H2.2: Todos los fondos: β₁ < 0 - Todos pierden afiliados en volatilidad
    
    Parameters:
        modelos_por_fondo (dict): Resultados por cada TipodeFondo
        alpha (float): Nivel de significancia
    
    Returns:
        dict: Resultados de los 2 tests
    """
    print(f"\n{'='*70}")
    print("TEST DE HIPÓTESIS - PANEL 2: FLIGHT-TO-QUALITY")
    print(f"{'='*70}")
    
    resultados = {}
    
    # Extraer coeficientes de PC1_Global_c por fondo
    coefs_por_fondo = {}
    for fondo, modelo in modelos_por_fondo.items():
        if 'PC1_Global_c' in modelo['coefs'].index:
            coefs_por_fondo[fondo] = {
                'coef': modelo['coefs']['PC1_Global_c'],
                'se': modelo['se']['PC1_Global_c'],
                'pval': modelo['pvals']['PC1_Global_c']
            }
    
    print("\n📊 Coeficientes β₁(PC1_Global_c) por Fondo:")
    for fondo in sorted(coefs_por_fondo.keys()):
        info = coefs_por_fondo[fondo]
        sig = '***' if info['pval'] < 0.01 else '**' if info['pval'] < 0.05 else '*' if info['pval'] < 0.10 else ''
        print(f"   Fondo {fondo}: β₁ = {info['coef']:7.4f} (SE={info['se']:.4f}, p={info['pval']:.4f}) {sig}")
    
    # H2.1: Flight-to-Quality (Fondo 0 más resiliente que Fondo 3)
    print("\n📊 H2.1: Flight-to-Quality Effect")
    print("   H0: β₁(Fondo0) ≤ β₁(Fondo3)")
    print("   HA: β₁(Fondo0) > β₁(Fondo3)")
    
    if 0 in coefs_por_fondo and 3 in coefs_por_fondo:
        beta_f0 = coefs_por_fondo[0]['coef']
        beta_f3 = coefs_por_fondo[3]['coef']
        
        # Test de diferencia (aproximado)
        # Usamos t-test con SE combinado
        se_f0 = coefs_por_fondo[0]['se']
        se_f3 = coefs_por_fondo[3]['se']
        
        diff = beta_f0 - beta_f3
        se_diff = np.sqrt(se_f0**2 + se_f3**2)
        t_diff = diff / se_diff
        pval_diff = 1 - stats.t.cdf(t_diff, df=100)  # Aproximación con df=100
        
        h21 = {
            'hipotesis': 'H2.1',
            'beta_f0': beta_f0,
            'beta_f3': beta_f3,
            'diferencia': diff,
            'se_diff': se_diff,
            't_stat': t_diff,
            'pval': pval_diff,
            'rechaza_h0': (diff > 0) and (pval_diff < alpha),
            'resultado': '✅ CONFIRMADA' if (diff > 0) and (pval_diff < alpha) else '❌ NO CONFIRMADA'
        }
        resultados['H2.1'] = h21
        
        print(f"   • β₁(Fondo0) = {beta_f0:.4f}")
        print(f"   • β₁(Fondo3) = {beta_f3:.4f}")
        print(f"   • Diferencia = {diff:.4f} (SE={se_diff:.4f})")
        print(f"   • t-stat = {t_diff:.3f}, p-valor = {pval_diff:.4f}")
        print(f"   • {h21['resultado']}")
        
        if diff > 0:
            print(f"   💡 Fondo 0 (conservador) ES más resiliente que Fondo 3 (agresivo)")
        else:
            print(f"   💡 NO hay diferencia significativa entre fondos")
    else:
        print("   ⚠️  No hay datos suficientes para Fondo 0 o Fondo 3")
        h21 = {'hipotesis': 'H2.1', 'resultado': '⚠️  NO TESTEABLE'}
        resultados['H2.1'] = h21
    
    # H2.2: Todos los fondos pierden afiliados (β₁ < 0)
    print("\n📊 H2.2: Todos los fondos reaccionan negativamente")
    print("   H0: Existe al menos un fondo con β₁ ≥ 0")
    print("   HA: Todos los fondos tienen β₁ < 0")
    
    todos_negativos = all([info['coef'] < 0 for info in coefs_por_fondo.values()])
    significativos = [fondo for fondo, info in coefs_por_fondo.items() if info['pval'] < 0.10]
    
    h22 = {
        'hipotesis': 'H2.2',
        'fondos_evaluados': len(coefs_por_fondo),
        'todos_negativos': todos_negativos,
        'significativos_negativos': len([f for f in significativos if coefs_por_fondo[f]['coef'] < 0]),
        'rechaza_h0': todos_negativos,
        'resultado': '✅ CONFIRMADA (todos β₁ < 0)' if todos_negativos else '❌ NO CONFIRMADA'
    }
    resultados['H2.2'] = h22
    
    print(f"   • Fondos evaluados: {len(coefs_por_fondo)}")
    print(f"   • Todos con β₁ < 0: {'SÍ' if todos_negativos else 'NO'}")
    print(f"   • Significativos (p<0.10): {len(significativos)}")
    print(f"   • {h22['resultado']}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN PANEL 2:")
    confirmadas = sum([1 for h in resultados.values() if h.get('rechaza_h0', False)])
    print(f"   Hipótesis confirmadas: {confirmadas}/2")
    print(f"{'='*70}")
    
    return resultados

# ==========================================
# FUNCIÓN: GENERAR TABLA RESUMEN
# ==========================================

def generar_tabla_resumen_hipotesis(resultados_p1, resultados_p2, nombre_archivo="resumen_hipotesis"):
    """
    Genera tabla resumen con todas las hipótesis testeadas.
    
    Parameters:
        resultados_p1 (dict): Resultados Panel 1
        resultados_p2 (dict): Resultados Panel 2
        nombre_archivo (str): Nombre del archivo
    
    Returns:
        DataFrame: Tabla resumen
    """
    print(f"\n{'='*70}")
    print("GENERANDO TABLA RESUMEN DE HIPÓTESIS")
    print(f"{'='*70}")
    
    filas = []
    
    # Panel 1
    for hip_id, resultado in resultados_p1.items():
        filas.append({
            'Panel': 'Panel 1: Aportes',
            'Hipótesis': hip_id,
            'Enunciado': resultado.get('decision', 'N/A'),
            'Coeficiente': f"{resultado['coeficiente']:.4f}" if 'coeficiente' in resultado else 'N/A',
            'p-valor': f"{resultado['pval']:.4f}" if 'pval' in resultado else 'N/A',
            'Significancia': resultado.get('sig', ''),
            'Resultado': '✅ Confirmada' if resultado['rechaza_h0'] else '❌ No confirmada'
        })
    
    # Panel 2
    for hip_id, resultado in resultados_p2.items():
        if hip_id == 'H2.1':
            filas.append({
                'Panel': 'Panel 2: Reasignación',
                'Hipótesis': hip_id,
                'Enunciado': 'β₁(Fondo0) > β₁(Fondo3)',
                'Coeficiente': f"Δ={resultado.get('diferencia', 0):.4f}",
                'p-valor': f"{resultado.get('pval', 1):.4f}",
                'Significancia': '**' if resultado.get('pval', 1) < 0.05 else '*' if resultado.get('pval', 1) < 0.10 else '',
                'Resultado': resultado['resultado']
            })
        elif hip_id == 'H2.2':
            filas.append({
                'Panel': 'Panel 2: Reasignación',
                'Hipótesis': hip_id,
                'Enunciado': 'Todos β₁ < 0',
                'Coeficiente': f"{resultado['fondos_evaluados']} fondos",
                'p-valor': 'N/A (test conjunto)',
                'Significancia': '',
                'Resultado': resultado['resultado']
            })
    
    df_resumen = pd.DataFrame(filas)
    
    # Guardar
    df_resumen.to_excel(f"{OUTPUT_DIR}/tablas/{nombre_archivo}.xlsx", index=False)
    print(f"\n💾 Tabla guardada: tablas/{nombre_archivo}.xlsx")
    
    # Mostrar en consola
    print(f"\n📊 TABLA RESUMEN DE HIPÓTESIS:\n")
    print(df_resumen.to_string(index=False))
    
    # Contar confirmadas
    total_confirmadas = df_resumen['Resultado'].str.contains('✅').sum()
    total_hipotesis = len(df_resumen)
    
    print(f"\n📈 BALANCE FINAL:")
    print(f"   • Total hipótesis: {total_hipotesis}")
    print(f"   • Confirmadas: {total_confirmadas}")
    print(f"   • No confirmadas: {total_hipotesis - total_confirmadas}")
    print(f"   • Tasa de confirmación: {100*total_confirmadas/total_hipotesis:.1f}%")
    
    return df_resumen

# ==========================================
# EJECUTAR TEST DE HIPÓTESIS
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FASE 4: TEST DE HIPÓTESIS DE INVESTIGACIÓN")
    print("="*70)
    
    # ==========================================
    # 1. PANEL 1: APORTES
    # ==========================================
    
    modelo_p1 = resultados_modelos['panel1']
    resultados_h_p1 = testear_hipotesis_panel1(modelo_p1, alpha=0.05)
    
    # ==========================================
    # 2. PANEL 2: REASIGNACIÓN
    # ==========================================
    
    modelos_p2_fondos = resultados_modelos['panel2_por_fondo']
    resultados_h_p2 = testear_hipotesis_panel2(modelos_p2_fondos, alpha=0.05)
    
    # ==========================================
    # 3. GENERAR TABLA RESUMEN
    # ==========================================
    
    tabla_resumen = generar_tabla_resumen_hipotesis(
        resultados_h_p1,
        resultados_h_p2,
        nombre_archivo="test_hipotesis_resultados"
    )
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    
    print("\n" + "="*70)
    print("✅ FASE 4 COMPLETADA")
    print("="*70)
    print("\n📊 TESTS DE HIPÓTESIS EJECUTADOS:")
    print(f"   ✅ Panel 1: 3 hipótesis testeadas")
    print(f"   ✅ Panel 2: 2 hipótesis testeadas")
    print(f"   ✅ Total: 5 hipótesis operacionales")
    
    print("\n📁 Archivo generado:")
    print("   • tablas/test_hipotesis_resultados.xlsx")
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Revisar tabla de resultados")
    print("   2. Interpretar hipótesis no confirmadas")
    print("   3. Preparar narrativa para discusión")
    
    print("\n📌 Próximo paso: Ejecutar modulo_fase5_documentacion.py")
    print("="*70)
