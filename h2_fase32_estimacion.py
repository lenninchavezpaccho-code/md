"""
H2: HETEROGENEIDAD POR TIPO DE FONDO
FASE 3.2: ESTIMACIÓN DEL MODELO
================================

Modelo a estimar:
  ΔAfiliados_std = α + β₁·PC1_Global + β₂·PC1_Sistematico +
                       β₃·Dummy_Fondo1 +
                       β₄·(PC1_Global × Fondo1) +
                       β₅·(PC1_Sistematico × Fondo1) +
                       γ·Controles + δ_mes + ε

Hipótesis:
  H0: β₄ = β₅ = 0 (no hay heterogeneidad)
  HA: β₄ ≠ 0 o β₅ ≠ 0 (sí hay heterogeneidad)

Predicción:
  β₄ > 0 (Fondo 1 más sensible a riesgo global)
  β₅ < 0 (Fondo 1 más reactivo a riesgo sistemático)

Autor: Sistema de Análisis Doctoral
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FILE_PANEL_2 = 'panel_2_reasignacion.xlsx'
FILE_PREDICTORES = 'dataset_final_interacciones.xlsx'
FILE_CONTROLES = 'variables_control_final.xlsx'

COL_FECHA = 'Fecha'
COL_FONDO = 'TipodeFondo'
COL_AFILIADOS = 'Afiliados_total'
COL_EXCLUSION = 'Dummy_Inicio_Fondo0'

OUTPUT_DIR = Path('h2_heterogeneidad_fondo')
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# CLASE ESTIMACIÓN H2
# =============================================================================
def convertir_a_json_serializable(obj):
    """Convierte objetos numpy/pandas a tipos nativos de Python"""
    if isinstance(obj, dict):
        return {k: convertir_a_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convertir_a_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [convertir_a_json_serializable(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and isinstance(obj, np.generic):
        return obj.item()
    return obj


def detectar_no_serializables(obj, path="root"):
    """Retorna rutas y tipos de valores que no son serializables en JSON"""
    problemas = []

    if isinstance(obj, dict):
        for clave, valor in obj.items():
            problemas.extend(detectar_no_serializables(valor, f"{path}.{clave}"))
        return problemas

    if isinstance(obj, list):
        for idx, valor in enumerate(obj):
            problemas.extend(detectar_no_serializables(valor, f"{path}[{idx}]"))
        return problemas

    if isinstance(obj, tuple):
        for idx, valor in enumerate(obj):
            problemas.extend(detectar_no_serializables(valor, f"{path}[{idx}]"))
        return problemas

    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        problemas.append((path, type(obj).__name__))

    return problemas
class EstimacionH2:
    """
    Estima modelo de heterogeneidad por Fondo 1
    según Campbell & Viceira (2002) y Merton (1973)
    """
    
    def __init__(self):
        self.df = None
        self.modelo = None
        self.resultados = {}
        
    def cargar_y_preparar_datos(self):
        """Carga datos y prepara variables (igual que Fase 3.1)"""
        print("="*80)
        print("H2 - FASE 3.2: ESTIMACIÓN DEL MODELO")
        print("="*80)
        
        # Cargar archivos
        print(f"\n📂 Cargando datos...")
        df_panel = pd.read_excel(FILE_PANEL_2)
        df_ivs = pd.read_excel(FILE_PREDICTORES)
        df_ctrl = pd.read_excel(FILE_CONTROLES)
        
        # Convertir fechas
        for df in [df_panel, df_ivs, df_ctrl]:
            df[COL_FECHA] = pd.to_datetime(df[COL_FECHA])
        
        # Filtro
        if COL_EXCLUSION in df_panel.columns:
            df_panel = df_panel[df_panel[COL_EXCLUSION] == 0].copy()
        
        # Merge
        df = df_panel.merge(df_ivs, on=COL_FECHA, how='inner')
        df = df.merge(df_ctrl, on=COL_FECHA, how='inner')
        
        # Ordenar
        df = df.sort_values([COL_FONDO, COL_FECHA])
        
        # Variable dependiente
        df['Variacion_Neta_Afiliados'] = df.groupby(COL_FONDO)[COL_AFILIADOS].diff()
        
        # VD estandarizada
        df['Variacion_Neta_std'] = (
            (df['Variacion_Neta_Afiliados'] - df['Variacion_Neta_Afiliados'].mean()) / 
            df['Variacion_Neta_Afiliados'].std()
        )
        
        # Centrar controles
        controles = ['Tasa_Referencia_BCRP', 'Inflacion_t_1', 
                     'PBI_Crecimiento_Interanual', 'Tipo_Cambio']
        
        for col in controles:
            col_c = f"{col}_c"
            if col_c not in df.columns:
                df[col_c] = df[col] - df[col].mean()
        
        # VARIABLES H2
        df['Dummy_Fondo1'] = (df[COL_FONDO] == 1).astype(int)
        df['PC1_Global_x_Fondo1'] = df['PC1_Global_c'] * df['Dummy_Fondo1']
        df['PC1_Sist_x_Fondo1'] = df['PC1_Sistematico_c'] * df['Dummy_Fondo1']
        
        # Dummies de mes
        df['Mes'] = df[COL_FECHA].dt.month
        for mes in range(2, 13):
            df[f'Mes_{mes}'] = (df['Mes'] == mes).astype(int)
        
        print(f"✓ Datos preparados: {len(df)} observaciones")
        
        self.df = df
        return df
    
    def estimar_modelo_h2(self):
        """Estima modelo principal de H2 con interacciones"""
        print("\n" + "="*80)
        print("ESTIMACIÓN: MODELO H2 CON INTERACCIONES")
        print("="*80)
        
        df = self.df
        
        # Variables del modelo
        vd = 'Variacion_Neta_std'  # Estandarizada
        
        vars_principales = [
            'PC1_Global_c',
            'PC1_Sistematico_c',
        #   'Dummy_Fondo1',
            'PC1_Global_x_Fondo1',  # β₄ (hipótesis principal)
            'PC1_Sist_x_Fondo1'     # β₅ (hipótesis principal)
        ]
        
        controles = [
            'Tasa_Referencia_BCRP_c',
            'Inflacion_t_1_c',
            'PBI_Crecimiento_Interanual_c',
            'Tipo_Cambio_c'
        ]
        
        dummies_mes = [f'Mes_{i}' for i in range(2, 13)]
        
        # Todas las variables independientes
        vars_x = vars_principales + controles + dummies_mes
        
        # Preparar datos para panel
        df_clean = df[[vd] + vars_x + [COL_FONDO, COL_FECHA]].dropna()
        
        print(f"\n📊 Especificación del modelo:")
        print(f"   • Variable dependiente: {vd}")
        print(f"   • Variables principales: {len(vars_principales)}")
        print(f"   • Controles: {len(controles)}")
        print(f"   • Dummies mes: {len(dummies_mes)}")
        print(f"   • Total regresores: {len(vars_x)}")
        print(f"   • Observaciones: {len(df_clean)}")
        
        # Configurar índice de panel
        df_clean = df_clean.set_index([COL_FONDO, COL_FECHA])
        
        # Separar Y y X
        y = df_clean[vd]
        X = df_clean[vars_x]
        
        # ESTIMAR CON PANELOLS + ENTITY EFFECTS
        print(f"\n⏳ Estimando modelo con efectos fijos por Fondo...")
        
        modelo = PanelOLS(
            y, 
            X, 
            entity_effects=True  # Efectos fijos por TipodeFondo
        ).fit(cov_type='robust')  # Errores robustos a heterocedasticidad
        
        print(f"✓ Modelo estimado exitosamente")
        
        self.modelo = modelo
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("RESULTADOS DEL MODELO H2")
        print("="*80)
        print(modelo.summary)
        
        return modelo
    
    def extraer_coeficientes_clave(self):
        """Extrae y presenta coeficientes de interés para H2"""
        print("\n" + "="*80)
        print("COEFICIENTES CLAVE PARA HIPÓTESIS H2")
        print("="*80)
        
        modelo = self.modelo
        
        # Coeficientes de interés
        vars_interes = [
            'PC1_Global_c',
            'PC1_Sistematico_c',
        #   'Dummy_Fondo1',
            'PC1_Global_x_Fondo1',  # β₄
            'PC1_Sist_x_Fondo1'     # β₅
        ]
        
        print(f"\n{'Variable':<30} {'Coef':<12} {'SE':<10} {'t-stat':<10} {'p-value':<12} {'Sig'}")
        print("-"*90)
        
        resultados_interes = {}
        
        for var in vars_interes:
            coef = modelo.params[var]
            se = modelo.std_errors[var]
            tstat = modelo.tstats[var]
            pval = modelo.pvalues[var]
            
            sig = ""
            if pval < 0.01:
                sig = "***"
            elif pval < 0.05:
                sig = "**"
            elif pval < 0.10:
                sig = "*"
            
            print(f"{var:<30} {coef:>11.4f} {se:>9.4f} {tstat:>9.3f} {pval:>11.4f} {sig:>3}")
            
            resultados_interes[var] = {
                'coef': float(coef),
                'se': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'significativo': bool(pval < 0.05)
            }
        
        print("\nNota: *** p<0.01, ** p<0.05, * p<0.10")
        
        # Interpretación de las interacciones
        print("\n" + "="*80)
        print("INTERPRETACIÓN DE INTERACCIONES (H2)")
        print("="*80)
        
        beta4 = resultados_interes['PC1_Global_x_Fondo1']
        beta5 = resultados_interes['PC1_Sist_x_Fondo1']
        
        print(f"\nβ₄ (PC1_Global × Fondo1): {beta4['coef']:.4f}")
        if beta4['significativo']:
            if beta4['coef'] > 0:
                print(f"  ✅ CONFIRMA H2a: Fondo 1 MÁS sensible a riesgo global (p={beta4['p_value']:.4f})")
            else:
                print(f"  ⚠️  Signo contrario a predicción (p={beta4['p_value']:.4f})")
        else:
            print(f"  ❌ NO significativo (p={beta4['p_value']:.4f})")
        
        print(f"\nβ₅ (PC1_Sistematico × Fondo1): {beta5['coef']:.4f}")
        if beta5['significativo']:
            if beta5['coef'] < 0:
                print(f"  ✅ CONFIRMA H2b: Fondo 1 MÁS reactivo a riesgo sistemático (p={beta5['p_value']:.4f})")
            else:
                print(f"  ⚠️  Signo contrario a predicción (p={beta5['p_value']:.4f})")
        else:
            print(f"  ❌ NO significativo (p={beta5['p_value']:.4f})")
        
        self.resultados['coeficientes_interes'] = resultados_interes
        
        return resultados_interes
    
    def test_wald_conjunto(self):
        """Test de Wald para significancia conjunta de las interacciones"""
        print("\n" + "="*80)
        print("TEST DE WALD: SIGNIFICANCIA CONJUNTA")
        print("="*80)
        
        modelo = self.modelo
        
        # Hipótesis nula: β₄ = β₅ = 0
        print(f"\nH0: β₄ = β₅ = 0 (no hay heterogeneidad)")
        print(f"HA: β₄ ≠ 0 o β₅ ≠ 0 (sí hay heterogeneidad)")
        
        # Crear matriz de restricciones
        param_names = list(modelo.params.index)
        num_params = len(param_names)
        
        vars_test = ['PC1_Global_x_Fondo1', 'PC1_Sist_x_Fondo1']
        
        R = np.zeros((2, num_params))
        for i, var_name in enumerate(vars_test):
            if var_name in param_names:
                j = param_names.index(var_name)
                R[i, j] = 1.0
        
        # Ejecutar test de Wald
        wald_result = modelo.wald_test(R)
        
        print(f"\n📊 Resultados del Test de Wald:")
        print(f"   • Estadístico F: {wald_result.stat:.4f}")
        print(f"   • p-valor: {wald_result.pval:.4f}")
        print(f"   • Grados de libertad: ({wald_result.df}, {wald_result.df_denom})")
        
        if wald_result.pval < 0.05:
            print(f"\n✅ RECHAZA H0 (p < 0.05): Las interacciones son conjuntamente significativas")
            print(f"   → Hay evidencia de HETEROGENEIDAD por Fondo 1")
        elif wald_result.pval < 0.10:
            print(f"\n⚠️  RECHAZA H0 marginalmente (p < 0.10)")
            print(f"   → Evidencia débil de heterogeneidad")
        else:
            print(f"\n❌ NO RECHAZA H0 (p ≥ 0.10): Las interacciones no son significativas")
            print(f"   → No hay evidencia suficiente de heterogeneidad")
        
        self.resultados['test_wald'] = {
            'f_stat': float(wald_result.stat),
            'p_value': float(wald_result.pval),
            'df': int(wald_result.df),
            'rechaza_h0': bool(wald_result.pval < 0.05)
        }
        
        return wald_result
    
    def calcular_efectos_totales(self):
        """Calcula efectos totales para Fondo 1 vs otros fondos"""
        print("\n" + "="*80)
        print("EFECTOS TOTALES: FONDO 1 vs OTROS FONDOS")
        print("="*80)
        
        modelo = self.modelo
        
        # Extraer coeficientes
        beta1 = modelo.params['PC1_Global_c']
        beta2 = modelo.params['PC1_Sistematico_c']
        beta4 = modelo.params['PC1_Global_x_Fondo1']
        beta5 = modelo.params['PC1_Sist_x_Fondo1']
        
        # Efectos para otros fondos (Fondo 0, 2, 3)
        efecto_global_otros = beta1
        efecto_sist_otros = beta2
        
        # Efectos para Fondo 1
        efecto_global_fondo1 = beta1 + beta4
        efecto_sist_fondo1 = beta2 + beta5
        
        print(f"\n📊 Efecto de PC1_Global (riesgo global):")
        print(f"   • Fondos 0, 2, 3: β₁ = {efecto_global_otros:.4f}")
        print(f"   • Fondo 1: β₁ + β₄ = {efecto_global_fondo1:.4f}")
        print(f"   • Diferencia: {efecto_global_fondo1 - efecto_global_otros:.4f}")
        
        print(f"\n📊 Efecto de PC1_Sistematico (riesgo sistemático):")
        print(f"   • Fondos 0, 2, 3: β₂ = {efecto_sist_otros:.4f}")
        print(f"   • Fondo 1: β₂ + β₅ = {efecto_sist_fondo1:.4f}")
        print(f"   • Diferencia: {efecto_sist_fondo1 - efecto_sist_otros:.4f}")
        
        self.resultados['efectos_totales'] = {
            'pc1_global': {
                'fondo_1': float(efecto_global_fondo1),
                'otros_fondos': float(efecto_global_otros),
                'diferencia': float(efecto_global_fondo1 - efecto_global_otros)
            },
            'pc1_sistematico': {
                'fondo_1': float(efecto_sist_fondo1),
                'otros_fondos': float(efecto_sist_otros),
                'diferencia': float(efecto_sist_fondo1 - efecto_sist_otros)
            }
        }
        
        return self.resultados['efectos_totales']
    
    def guardar_resultados(self):
        """Guarda todos los resultados en archivos"""
        print("\n" + "="*80)
        print("GUARDANDO RESULTADOS")
        print("="*80)
        
        # 1. Resumen del modelo (texto)
        archivo_summary = OUTPUT_DIR / 'h2_modelo_summary.txt'
        with open(archivo_summary, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MODELO H2: HETEROGENEIDAD POR FONDO 1\n")
            f.write("="*80 + "\n\n")
            f.write(self.modelo.summary.as_text())
        print(f"✓ {archivo_summary}")
        
        # 2. Guardar resultados estructurados en JSON
        archivo_json = OUTPUT_DIR / 'h2_resultados.json'
        resultados_serializables = convertir_a_json_serializable(self.resultados)

        problemas_serializacion = detectar_no_serializables(resultados_serializables)
        if problemas_serializacion:
            print("⚠️  Valores no serializables detectados (conversión a cadena):")
            for ruta, tipo in problemas_serializacion:
                print(f"   • {ruta}: {tipo}")

        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(resultados_serializables, f, ensure_ascii=False, indent=4)
        print(f"✓ {archivo_json} (formato JSON)")

        # 3. Guardar con pickle (para depuración)
        import pickle
        archivo_pickle = OUTPUT_DIR / 'h2_resultados.pkl'
        with open(archivo_pickle, 'wb') as f:
            pickle.dump(self.resultados, f)
        print(f"✓ {archivo_pickle} (formato pickle)")

        # 4. Tabla de coeficientes (Excel)
        archivo_excel = OUTPUT_DIR / 'h2_coeficientes.xlsx'

        df_coefs = pd.DataFrame({
            'Variable': list(self.modelo.params.index),
            'Coeficiente': [float(x) for x in self.modelo.params.values],
            'SE': [float(x) for x in self.modelo.std_errors.values],
            't_stat': [float(x) for x in self.modelo.tstats.values],
            'p_value': [float(x) for x in self.modelo.pvalues.values],
            'CI_lower': [float(x) for x in self.modelo.conf_int()[0].values],
            'CI_upper': [float(x) for x in self.modelo.conf_int()[1].values]
        })
        
        df_coefs.to_excel(archivo_excel, index=False)
        print(f"✓ {archivo_excel}")
        
        print(f"\n✅ Resultados guardados en: {OUTPUT_DIR}/")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def ejecutar_estimacion_h2():
    """Ejecuta estimación completa de H2"""
    
    estimador = EstimacionH2()
    
    try:
        # Paso 1: Cargar datos
        estimador.cargar_y_preparar_datos()
        
        # Paso 2: Estimar modelo
        estimador.estimar_modelo_h2()
        
        # Paso 3: Extraer coeficientes clave
        estimador.extraer_coeficientes_clave()
        
        # Paso 4: Test de Wald
        estimador.test_wald_conjunto()
        
        # Paso 5: Calcular efectos totales
        estimador.calcular_efectos_totales()
        
        # Paso 6: Guardar resultados
        estimador.guardar_resultados()
        
        print("\n" + "="*80)
        print("✅ FASE 3.2 COMPLETADA: MODELO H2 ESTIMADO")
        print("="*80)
        
        return estimador
        
    except Exception as e:
        print(f"\n❌ ERROR durante estimación:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    estimador = ejecutar_estimacion_h2()
