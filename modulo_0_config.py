"""
═══════════════════════════════════════════════════════════════════════════════
MÓDULO 0: CONFIGURACIÓN E IMPORTACIONES
═══════════════════════════════════════════════════════════════════════════════

Este módulo centraliza todas las configuraciones, rutas de archivos y 
preparación inicial de datos para el análisis doctoral.

Exporta:
  - CONFIG: Clase con toda la configuración del proyecto
  - OUTPUT_DIR: Directorio de salida para resultados
  - datos_raw: Diccionario con datos crudos cargados
  - df_p1: DataFrame Panel 1 preparado con MultiIndex
  - df_p2: DataFrame Panel 2 preparado con MultiIndex
  - ctrl_c_p1: Lista de nombres de controles centrados Panel 1
  - ctrl_c_p2: Lista de nombres de controles centrados Panel 2
  - mes_p1: Lista de nombres de dummies de mes Panel 1
  - mes_p2: Lista de nombres de dummies de mes Panel 2

═══════════════════════════════════════════════════════════════════════════════
"""

# ==========================================
# IMPORTACIONES NECESARIAS
# ==========================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# Librerías de econometría
try:
    from linearmodels.panel import PanelOLS, RandomEffects, compare
    import statsmodels.api as sm
    from scipy import stats
    print("✅ Librerías de econometría cargadas correctamente")
except ImportError as e:
    print(f"⚠️ ERROR: Falta instalar librerías necesarias")
    print(f"   Ejecuta: pip install linearmodels statsmodels scipy")
    raise

# ==========================================
# CLASE DE CONFIGURACIÓN
# ==========================================

class ConfigTesis:
    """
    Clase que centraliza toda la configuración del proyecto.

    Incluye:
      - Rutas de archivos de entrada
      - Nombres de columnas clave
      - Variables de exclusión
      - Variables predictoras e interacciones
      - Variables de control
    """

    # ========== ARCHIVOS DE ENTRADA ==========
    FILE_PANEL_1 = 'panel_1_aportes.xlsx'
    FILE_PANEL_2 = 'panel_2_reasignacion.xlsx'
    FILE_PANEL_3 = 'panel_3_portafolio.xlsx'
    FILE_PREDICTORES = 'dataset_final_interacciones.xlsx'
    FILE_CONTROLES = 'variables_control_final.xlsx'

    # ========== NOMBRES DE COLUMNAS ==========
    # Columnas comunes
    COL_FECHA = 'Fecha'

    # Panel 1: Aportes
    COL_AFP = 'AFP'
    VAR_Y_PANEL1 = 'ln_Aportes_AFP'
    DUMMY_EXCLUSION_P1 = 'Dummy_Ajuste_Aportes_Sep2013'

    # Panel 2: Reasignación
    COL_FONDO = 'TipodeFondo'
    VAR_Y_PANEL2 = 'Variacion_Neta_Afiliados'
    DUMMY_EXCLUSION_P2 = 'Dummy_Inicio_Fondo0'

    # Panel 3: Portafolio
    COL_FONDO_P3 = 'Fondo'
    COL_EMISOR = 'Emisor_Origen'
    COL_SECTOR = 'Sector'
    VAR_Y_PANEL3 = 'Stock_%'
    DUMMY_EXCLUSION_P3 = 'Dummy_Inicio_Fondo0'

    # ========== VARIABLES PREDICTORAS E INTERACCIONES ==========
    VARS_IV_MOD = [
        'PC1_Global_c',
        'PC1_Sistematico_c',
        'D_COVID',
        'Int_Global_COVID',
        'Int_Sistematico_COVID'
    ]

    # ========== VARIABLES DE CONTROL ==========
    VARS_CONTROL = [
        'Tasa_Referencia_BCRP',
        'Inflacion_t_1',
        'PBI_Crecimiento_Interanual',
        'Tipo_Cambio'
    ]


# ==========================================
# CREAR CONFIGURACIÓN GLOBAL
# ==========================================

CONFIG = ConfigTesis()
print("\n" + "="*70)
print("CONFIGURACIÓN DEL PROYECTO CARGADA")
print("="*70)
print(f"  Panel 1: {CONFIG.FILE_PANEL_1}")
print(f"  Panel 2: {CONFIG.FILE_PANEL_2}")
print(f"  Panel 3: {CONFIG.FILE_PANEL_3}")
print(f"  Predictores: {CONFIG.FILE_PREDICTORES}")
print(f"  Controles: {CONFIG.FILE_CONTROLES}")
print("="*70)


# ==========================================
# CREAR ESTRUCTURA DE CARPETAS
# ==========================================

def crear_estructura_carpetas():
    """
    Crea la estructura de carpetas para guardar resultados.

    Returns:
        str: Ruta del directorio principal de salida
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"resultados_tesis_{timestamp}"

    # Crear carpetas
    carpetas = [
        output_dir,
        f"{output_dir}/diagnosticos",
        f"{output_dir}/tablas",
        f"{output_dir}/graficos",
        f"{output_dir}/robustez",
        f"{output_dir}/panel3"
    ]

    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)

    print(f"\n✅ Estructura de carpetas creada: {output_dir}")
    return output_dir


OUTPUT_DIR = crear_estructura_carpetas()


# ==========================================
# CARGAR DATOS
# ==========================================

def cargar_datos():
    """
    Carga todos los archivos Excel necesarios.

    Returns:
        dict: Diccionario con todos los DataFrames crudos
    """
    print(f"\n{'='*70}")
    print("CARGANDO ARCHIVOS DE DATOS")
    print(f"{'='*70}")

    datos = {}

    try:
        # Cargar cada archivo
        datos['panel1'] = pd.read_excel(CONFIG.FILE_PANEL_1)
        print(f"✅ Panel 1 cargado: {datos['panel1'].shape}")

        datos['panel2'] = pd.read_excel(CONFIG.FILE_PANEL_2)
        print(f"✅ Panel 2 cargado: {datos['panel2'].shape}")

        datos['panel3'] = pd.read_excel(CONFIG.FILE_PANEL_3)
        print(f"✅ Panel 3 cargado: {datos['panel3'].shape}")

        datos['predictores'] = pd.read_excel(CONFIG.FILE_PREDICTORES)
        print(f"✅ Predictores cargados: {datos['predictores'].shape}")

        datos['controles'] = pd.read_excel(CONFIG.FILE_CONTROLES)
        print(f"✅ Controles cargados: {datos['controles'].shape}")

        # Convertir fechas
        for key in datos:
            if CONFIG.COL_FECHA in datos[key].columns:
                datos[key][CONFIG.COL_FECHA] = pd.to_datetime(
                    datos[key][CONFIG.COL_FECHA], 
                    dayfirst=False
                )

        print(f"✅ Todas las fechas convertidas a datetime")
        print(f"{'='*70}")

        return datos

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: No se encontró el archivo: {e.filename}")
        print(f"\n📁 Archivos en el directorio actual:")
        for archivo in os.listdir('.'):
            if archivo.endswith('.xlsx'):
                print(f"   • {archivo}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR al cargar datos: {e}")
        import traceback
        traceback.print_exc()
        raise


datos_raw = cargar_datos()


# ==========================================
# PREPARAR PANEL 1 (APORTES)
# ==========================================

def preparar_panel1(datos):
    """
    Prepara DataFrame para Panel 1 (Aportes).

    Operaciones:
      1. Crear ln_Aportes_AFP si no existe (desde Aportes_total)
      2. Filtrar por dummy de exclusión
      3. Fusionar con predictores y controles
      4. Crear dummies de mes
      5. Centrar controles
      6. Crear interacciones de control con COVID

    Returns:
        tuple: (df_p1, ctrl_c_p1, mes_p1)
    """
    print(f"\n{'='*70}")
    print("PREPARANDO PANEL 1 (APORTES)")
    print(f"{'='*70}")

    # 0. Crear ln_Aportes_AFP si no existe
    df = datos['panel1'].copy()
    if CONFIG.VAR_Y_PANEL1 not in df.columns:
        if 'Aportes_total' in df.columns:
            print(f"⚠️  '{CONFIG.VAR_Y_PANEL1}' no encontrada. Creando desde 'Aportes_total'...")
            # Convertir valores no positivos a NaN
            df['Aportes_total'] = df['Aportes_total'].apply(
                lambda x: x if (pd.notna(x) and x > 0) else np.nan
            )
            df[CONFIG.VAR_Y_PANEL1] = np.log(df['Aportes_total'])
            print(f"✅ {CONFIG.VAR_Y_PANEL1} creada correctamente")
        else:
            raise KeyError(f"No se encontró ni '{CONFIG.VAR_Y_PANEL1}' ni 'Aportes_total' en Panel 1")

    # 1. Filtrar por dummy de exclusión
    if CONFIG.DUMMY_EXCLUSION_P1 in df.columns:
        obs_antes = len(df)
        df = df[df[CONFIG.DUMMY_EXCLUSION_P1] == 0].copy()
        print(f"✅ Filtrado aplicado: {obs_antes - len(df)} obs. eliminadas")

    # 2. Fusionar con predictores y controles
    df = pd.merge(df, datos['predictores'], on=CONFIG.COL_FECHA, how='inner')
    df = pd.merge(df, datos['controles'], on=CONFIG.COL_FECHA, how='inner')
    print(f"✅ Datos fusionados: {len(df)} observaciones")

    # 3. Crear dummies de mes
    df['month'] = df[CONFIG.COL_FECHA].dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='Mes', drop_first=True, dtype=int)
    df = pd.concat([df, month_dummies], axis=1)
    mes_p1 = month_dummies.columns.tolist()
    print(f"✅ Dummies de mes creadas: {len(mes_p1)}")

    # 4. Centrar controles
    ctrl_c_p1 = []
    for col in CONFIG.VARS_CONTROL:
        col_c = f"{col}_c"
        df[col_c] = df[col] - df[col].mean()
        ctrl_c_p1.append(col_c)
    print(f"✅ Controles centrados: {len(ctrl_c_p1)}")

    # 5. Crear interacciones de control con COVID
    ctrl_int_p1 = []
    for col_c in ctrl_c_p1:
        int_name = f"Int_{col_c}_COVID"
        df[int_name] = df[col_c] * df['D_COVID']
        ctrl_int_p1.append(int_name)
    print(f"✅ Interacciones creadas: {len(ctrl_int_p1)}")

    # 6. Eliminar NAs
    vars_necesarias = ([CONFIG.VAR_Y_PANEL1] + CONFIG.VARS_IV_MOD + 
                       ctrl_c_p1 + ctrl_int_p1 + mes_p1 + 
                       [CONFIG.COL_AFP, CONFIG.COL_FECHA])
    obs_antes_na = len(df)
    df = df.dropna(subset=vars_necesarias)
    print(f"✅ NAs eliminados: {obs_antes_na - len(df)} obs. removidas, {len(df)} finales")

    print(f"{'='*70}")

    return df, ctrl_c_p1, mes_p1


df_p1, ctrl_c_p1, mes_p1 = preparar_panel1(datos_raw)


# ==========================================
# PREPARAR PANEL 2 (REASIGNACIÓN)
# ==========================================

def preparar_panel2(datos):
    """
    Prepara DataFrame para Panel 2 (Reasignación).

    Operaciones:
      1. Crear entidad compuesta AFP_Fondo
      2. Filtrar por dummy de exclusión
      3. Fusionar con predictores y controles
      4. Crear dummies de mes
      5. Centrar controles
      6. Crear interacciones de control con COVID

    Returns:
        tuple: (df_p2, ctrl_c_p2, mes_p2)
    """
    print(f"\n{'='*70}")
    print("PREPARANDO PANEL 2 (REASIGNACIÓN)")
    print(f"{'='*70}")

    # 1. Crear entidad compuesta
    df = datos['panel2'].copy()
    df['Entidad_Compuesta'] = (
        df[CONFIG.COL_AFP].astype(str) + '_F' + df[CONFIG.COL_FONDO].astype(str)
    )
    print(f"✅ Entidad compuesta creada: {df['Entidad_Compuesta'].nunique()} entidades")

    # 2. Filtrar por dummy de exclusión
    if CONFIG.DUMMY_EXCLUSION_P2 in df.columns:
        obs_antes = len(df)
        df = df[df[CONFIG.DUMMY_EXCLUSION_P2] == 0].copy()
        print(f"✅ Filtrado aplicado: {obs_antes - len(df)} obs. eliminadas")

    # 3. Fusionar con predictores y controles
    df = pd.merge(df, datos['predictores'], on=CONFIG.COL_FECHA, how='inner')
    df = pd.merge(df, datos['controles'], on=CONFIG.COL_FECHA, how='inner')
    print(f"✅ Datos fusionados: {len(df)} observaciones")

    # 4. Crear dummies de mes
    df['month'] = df[CONFIG.COL_FECHA].dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='Mes', drop_first=True, dtype=int)
    df = pd.concat([df, month_dummies], axis=1)
    mes_p2 = month_dummies.columns.tolist()
    print(f"✅ Dummies de mes creadas: {len(mes_p2)}")

    # 5. Centrar controles
    ctrl_c_p2 = []
    for col in CONFIG.VARS_CONTROL:
        col_c = f"{col}_c"
        df[col_c] = df[col] - df[col].mean()
        ctrl_c_p2.append(col_c)
    print(f"✅ Controles centrados: {len(ctrl_c_p2)}")

    # 6. Crear interacciones de control con COVID
    ctrl_int_p2 = []
    for col_c in ctrl_c_p2:
        int_name = f"Int_{col_c}_COVID"
        df[int_name] = df[col_c] * df['D_COVID']
        ctrl_int_p2.append(int_name)
    print(f"✅ Interacciones creadas: {len(ctrl_int_p2)}")

    # 7. Eliminar NAs
    vars_necesarias = ([CONFIG.VAR_Y_PANEL2] + CONFIG.VARS_IV_MOD + 
                       ctrl_c_p2 + ctrl_int_p2 + mes_p2 + 
                       ['Entidad_Compuesta', CONFIG.COL_FECHA])
    obs_antes_na = len(df)
    df = df.dropna(subset=vars_necesarias)
    print(f"✅ NAs eliminados: {obs_antes_na - len(df)} obs. removidas, {len(df)} finales")

    print(f"{'='*70}")

    return df, ctrl_c_p2, mes_p2


df_p2, ctrl_c_p2, mes_p2 = preparar_panel2(datos_raw)


# ==========================================
# RESUMEN FINAL
# ==========================================

print(f"\n{'='*70}")
print("✅ MÓDULO 0 COMPLETADO")
print(f"{'='*70}")
print(f"\n📊 DATOS PREPARADOS:")
print(f"\n  Panel 1 (Aportes):")
print(f"    • Observaciones: {len(df_p1)}")
print(f"    • Entidades: {df_p1[CONFIG.COL_AFP].nunique()}")
print(f"    • Variable Y: {CONFIG.VAR_Y_PANEL1}")
print(f"\n  Panel 2 (Reasignación):")
print(f"    • Observaciones: {len(df_p2)}")
print(f"    • Entidades compuestas: {df_p2['Entidad_Compuesta'].nunique()}")
print(f"    • Variable Y: {CONFIG.VAR_Y_PANEL2}")
print(f"\n📁 Directorio de salida: {OUTPUT_DIR}")
print(f"\n📌 Variables exportadas:")
print(f"  • CONFIG")
print(f"  • OUTPUT_DIR")
print(f"  • datos_raw")
print(f"  • df_p1, ctrl_c_p1, mes_p1")
print(f"  • df_p2, ctrl_c_p2, mes_p2")
print(f"\n📌 Próximo paso: Ejecutar modulo_fase31_verificacion.py")
print(f"{'='*70}")


# ==========================================
# EJECUCIÓN DIRECTA
# ==========================================

if __name__ == "__main__":
    print("\n✅ Módulo 0 ejecutado correctamente como script principal")
    print("\n💡 Para usar en otros módulos:")
    print("   from modulo_0_config import CONFIG, OUTPUT_DIR, df_p1, df_p2, ...")
