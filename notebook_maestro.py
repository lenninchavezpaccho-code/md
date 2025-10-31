"""
═══════════════════════════════════════════════════════════════════════════════
NOTEBOOK MAESTRO: ANÁLISIS DOCTORAL COMPLETO
═══════════════════════════════════════════════════════════════════════════════
Este notebook ejecuta secuencialmente todas las fases del análisis doctoral.

INSTRUCCIONES:
1. Asegúrate de que todos los archivos .py de módulos estén en el mismo directorio
2. Asegúrate de que los archivos Excel estén en el directorio
3. Ejecuta este archivo para correr el análisis completo

Módulos incluidos:
- Módulo 0: Configuración e Importaciones
- Módulo 3.1: Verificación Pre-Estimación
- Módulo 3.2: Estimación de Modelos Finales
- Módulo 3.3: Validación de Supuestos (siguiente)
- Módulo 3.4: Análisis de Robustez (siguiente)
- Módulo 4: Test de Hipótesis (siguiente)
- Módulo 5: Documentación Final (siguiente)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
from datetime import datetime
import traceback

# ==========================================
# FUNCIÓN: EJECUTAR MÓDULO
# ==========================================

def ejecutar_modulo(nombre_modulo, descripcion):
    """
    Ejecuta un módulo de análisis y captura errores.
    
    Parameters:
        nombre_modulo (str): Nombre del archivo .py
        descripcion (str): Descripción del módulo
    
    Returns:
        bool: True si exitoso, False si falló
    """
    print("\n" + "="*80)
    print(f"🚀 EJECUTANDO: {descripcion}")
    print(f"📄 Módulo: {nombre_modulo}")
    print("="*80)
    
    try:
        # Ejecutar módulo
        with open(nombre_modulo, 'r', encoding='utf-8') as f:
            codigo = f.read()
        
        # Ejecutar en el namespace global actual
        exec(codigo, globals())
        
        print("\n" + "="*80)
        print(f"✅ {descripcion} - COMPLETADO")
        print("="*80)
        
        return True
        
    except FileNotFoundError:
        print("\n" + "="*80)
        print(f"❌ ERROR: Archivo no encontrado - {nombre_modulo}")
        print("="*80)
        print("\nAsegúrate de que el archivo esté en el mismo directorio.")
        return False
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR EN {descripcion}")
        print("="*80)
        print(f"\nTipo de error: {type(e).__name__}")
        print(f"Mensaje: {e}")
        print("\nTraceback completo:")
        traceback.print_exc()
        return False

# ==========================================
# FUNCIÓN: GENERAR REPORTE FINAL
# ==========================================

def generar_reporte_final(output_dir):
    """
    Genera un reporte consolidado de todo el análisis.
    """
    print("\n" + "="*80)
    print("📊 GENERANDO REPORTE FINAL")
    print("="*80)
    
    reporte_file = f"{output_dir}/REPORTE_FINAL.txt"
    
    with open(reporte_file, 'w', encoding='utf-8') as f:
        f.write("═"*80 + "\n")
        f.write("REPORTE FINAL - ANÁLISIS DOCTORAL\n")
        f.write("Sistema Privado de Pensiones Peruano\n")
        f.write("═"*80 + "\n\n")
        f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("═"*80 + "\n")
        f.write("ESTRUCTURA DE RESULTADOS\n")
        f.write("═"*80 + "\n\n")
        
        # Listar archivos generados
        f.write("📁 CARPETA: diagnosticos/\n")
        diagnosticos_dir = f"{output_dir}/diagnosticos"
        if os.path.exists(diagnosticos_dir):
            for archivo in os.listdir(diagnosticos_dir):
                f.write(f"   • {archivo}\n")
        
        f.write("\n📁 CARPETA: tablas/\n")
        tablas_dir = f"{output_dir}/tablas"
        if os.path.exists(tablas_dir):
            for archivo in os.listdir(tablas_dir):
                f.write(f"   • {archivo}\n")
        
        f.write("\n📁 CARPETA: graficos/\n")
        graficos_dir = f"{output_dir}/graficos"
        if os.path.exists(graficos_dir):
            for archivo in os.listdir(graficos_dir):
                f.write(f"   • {archivo}\n")
        
        f.write("\n\n═"*80 + "\n")
        f.write("PRÓXIMOS PASOS\n")
        f.write("═"*80 + "\n\n")
        f.write("1. Revisar los diagnósticos en la carpeta 'diagnosticos/'\n")
        f.write("2. Analizar las tablas de coeficientes en 'tablas/'\n")
        f.write("3. Revisar los gráficos en 'graficos/'\n")
        f.write("4. Ejecutar análisis de robustez (Fase 3.4)\n")
        f.write("5. Testear hipótesis formalmente (Fase 4)\n")
        f.write("6. Generar documentación final (Fase 5)\n")
    
    print(f"✅ Reporte final guardado: {reporte_file}")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    
    print("\n\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  ANÁLISIS ECONOMÉTRICO DOCTORAL - SISTEMA DE PENSIONES PERUANO".center(78) + "║")
    print("║" + " "*78 + "║")
    print("║" + "  Fases 3-5: Análisis Completo".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    hora_inicio = datetime.now()
    print(f"\n⏰ Hora de inicio: {hora_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Lista de módulos a ejecutar
    modulos = [
        {
            'archivo': 'modulo_0_config.py',
            'descripcion': 'FASE 0: Configuración e Importaciones',
            'obligatorio': True
        },
        {
            'archivo': 'modulo_fase31_verificacion.py',
            'descripcion': 'FASE 3.1: Verificación Pre-Estimación',
            'obligatorio': True
        },
        {
            'archivo': 'modulo_fase32_estimacion.py',
            'descripcion': 'FASE 3.2: Estimación de Modelos Finales',
            'obligatorio': True
        }
    ]
    
    # Ejecutar módulos secuencialmente
    exitos = 0
    fallos = 0
    
    for modulo in modulos:
        exito = ejecutar_modulo(modulo['archivo'], modulo['descripcion'])
        
        if exito:
            exitos += 1
        else:
            fallos += 1
            if modulo['obligatorio']:
                print("\n" + "⚠️ "*20)
                print("⚠️  MÓDULO OBLIGATORIO FALLÓ - DETENIENDO EJECUCIÓN")
                print("⚠️ "*20)
                break
    
    # Generar reporte final
    if exitos > 0:
        try:
            generar_reporte_final(OUTPUT_DIR)
        except:
            print("⚠️  No se pudo generar reporte final (OUTPUT_DIR no disponible)")
    
    # Resumen final
    hora_fin = datetime.now()
    duracion = (hora_fin - hora_inicio).total_seconds()
    
    print("\n\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  RESUMEN DE EJECUCIÓN".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    print(f"\n⏰ Hora de inicio:  {hora_inicio.strftime('%H:%M:%S')}")
    print(f"⏰ Hora de fin:     {hora_fin.strftime('%H:%M:%S')}")
    print(f"⏱️  Duración total:  {duracion:.1f} segundos ({duracion/60:.1f} minutos)")
    
    print(f"\n📊 Módulos ejecutados:")
    print(f"   ✅ Exitosos: {exitos}")
    print(f"   ❌ Fallidos: {fallos}")
    print(f"   📝 Total:    {exitos + fallos}")
    
    if fallos == 0:
        print("\n" + "🎉 "*20)
        print("🎉  ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print("🎉 "*20)
        print(f"\n📁 Revisa los resultados en: {OUTPUT_DIR}")
    else:
        print("\n" + "⚠️ "*20)
        print("⚠️  ANÁLISIS INCOMPLETO - Revisa los errores arriba")
        print("⚠️ "*20)
    
    print("\n" + "="*80)
    print("FIN DE LA EJECUCIÓN")
    print("="*80 + "\n")


# ==========================================
# INSTRUCCIONES DE USO
# ==========================================

"""
═══════════════════════════════════════════════════════════════════════════════
INSTRUCCIONES DE USO
═══════════════════════════════════════════════════════════════════════════════

OPCIÓN 1: EJECUTAR TODO AUTOMÁTICAMENTE
----------------------------------------
1. Guarda este archivo como: notebook_maestro.py
2. Guarda los otros módulos en el mismo directorio:
   - modulo_0_config.py
   - modulo_fase31_verificacion.py
   - modulo_fase32_estimacion.py
3. Asegúrate de que los archivos Excel estén en el mismo directorio
4. Ejecuta en terminal:
   python notebook_maestro.py

OPCIÓN 2: EJECUTAR EN JUPYTER NOTEBOOK
---------------------------------------
1. Crea un nuevo notebook
2. En la primera celda, copia TODO el contenido de modulo_0_config.py
3. En la segunda celda, copia el contenido de modulo_fase31_verificacion.py
4. En la tercera celda, copia el contenido de modulo_fase32_estimacion.py
5. Ejecuta las celdas secuencialmente

OPCIÓN 3: EJECUTAR MÓDULOS INDIVIDUALES
----------------------------------------
1. Primero ejecuta: python modulo_0_config.py
2. Luego ejecuta: python modulo_fase31_verificacion.py
3. Luego ejecuta: python modulo_fase32_estimacion.py

ARCHIVOS NECESARIOS EN EL DIRECTORIO:
--------------------------------------
✅ panel_1_aportes.xlsx
✅ panel_2_reasignacion.xlsx
✅ panel_3_portafolio.xlsx
✅ dataset_final_interacciones.xlsx
✅ variables_control_final.xlsx

ESTRUCTURA DE SALIDA:
----------------------
resultados_tesis_YYYYMMDD_HHMMSS/
├── diagnosticos/
│   ├── verificacion_varianza_panel1.xlsx
│   ├── verificacion_varianza_panel2.xlsx
│   ├── validacion_entidades_panel2.xlsx
│   ├── hausman_panel1.txt
│   └── hausman_panel2.txt
├── tablas/
│   ├── modelo_panel_1_aportes.txt
│   ├── coeficientes_panel_1_aportes.xlsx
│   ├── modelo_panel_2_reasignacion_agregado.txt
│   └── tabla_comparativa_panel2.xlsx
├── graficos/
│   └── (se generarán en fases posteriores)
└── REPORTE_FINAL.txt

═══════════════════════════════════════════════════════════════════════════════
"""
