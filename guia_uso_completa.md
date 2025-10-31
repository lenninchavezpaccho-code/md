# 📚 GUÍA COMPLETA: ANÁLISIS ECONOMÉTRICO DOCTORAL

## Sistema Privado de Pensiones Peruano - Fases 3-5

---

## 📋 ÍNDICE

1. [Estructura del Proyecto](#estructura)
2. [Archivos Requeridos](#archivos)
3. [Módulos Disponibles](#modulos)
4. [Instrucciones de Uso](#instrucciones)
5. [Resultados Esperados](#resultados)
6. [Solución de Problemas](#problemas)

---

## 🗂️ ESTRUCTURA DEL PROYECTO <a name="estructura"></a>

```
tu_directorio/
│
├── 📄 ARCHIVOS DE CÓDIGO PYTHON
│   ├── modulo_0_config.py               [✅ DISPONIBLE]
│   ├── modulo_fase31_verificacion.py    [✅ DISPONIBLE]
│   ├── modulo_fase32_estimacion.py      [✅ DISPONIBLE]
│   ├── modulo_fase33_supuestos.py       [⏳ PRÓXIMO]
│   ├── modulo_fase34_robustez.py        [⏳ PRÓXIMO]
│   ├── modulo_fase4_hipotesis.py        [⏳ PRÓXIMO]
│   ├── modulo_fase5_documentacion.py    [⏳ PRÓXIMO]
│   └── notebook_maestro.py              [✅ DISPONIBLE]
│
├── 📊 ARCHIVOS DE DATOS (Excel)
│   ├── panel_1_aportes.xlsx
│   ├── panel_2_reasignacion.xlsx
│   ├── panel_3_portafolio.xlsx
│   ├── dataset_final_interacciones.xlsx
│   └── variables_control_final.xlsx
│
└── 📁 CARPETA DE RESULTADOS (se crea automáticamente)
    └── resultados_tesis_YYYYMMDD_HHMMSS/
        ├── diagnosticos/
        ├── tablas/
        ├── graficos/
        ├── robustez/
        └── panel3/
```

---

## 📂 ARCHIVOS REQUERIDOS <a name="archivos"></a>

### Archivos Excel Obligatorios

Asegúrate de tener estos 5 archivos Excel en el mismo directorio que los scripts:

| Archivo | Descripción | Columnas Clave |
|---------|-------------|----------------|
| `panel_1_aportes.xlsx` | Panel de aportes mensuales por AFP | `Fecha`, `AFP`, `ln_Aportes_AFP`, `Dummy_Ajuste_Aportes_Sep2013` |
| `panel_2_reasignacion.xlsx` | Panel de flujos de afiliados | `Fecha`, `AFP`, `TipodeFondo`, `Variacion_Neta_Afiliados`, `Dummy_Inicio_Fondo0` |
| `panel_3_portafolio.xlsx` | Panel de composición de portafolio | `Fecha`, `Fondo`, `Emisor_Origen`, `Sector`, `Stock_%`, `Dummy_Inicio_Fondo0` |
| `dataset_final_interacciones.xlsx` | Variables predictoras e interacciones | `Fecha`, `PC1_Global_c`, `PC1_Sistematico_c`, `D_COVID`, `Int_Global_COVID`, `Int_Sistematico_COVID` |
| `variables_control_final.xlsx` | Variables macroeconómicas de control | `Fecha`, `Tasa_Referencia_BCRP`, `Inflacion_t_1`, `PBI_Crecimiento_Interanual`, `Tipo_Cambio` |

---

## 🔧 MÓDULOS DISPONIBLES <a name="modulos"></a>

### ✅ Módulo 0: Configuración e Importaciones
**Archivo:** `modulo_0_config.py`

**Funciones:**
- Importar librerías necesarias
- Definir configuración global (clase `ConfigTesis`)
- Crear estructura de carpetas de resultados
- Cargar todos los archivos Excel
- Preparar datos (fusionar, crear dummies, centrar variables)

**Salida:**
- Variables globales: `CONFIG`, `OUTPUT_DIR`, `datos_raw`, `df_p1`, `df_p2`
- Carpetas creadas en `resultados_tesis_YYYYMMDD_HHMMSS/`

---

### ✅ Módulo Fase 3.1: Verificación Pre-Estimación
**Archivo:** `modulo_fase31_verificacion.py`

**Objetivos (Kerlinger 1986):**
1. Verificar varianza suficiente en todas las variables
2. Validar entidades sin varianza nula en Panel 2
3. Test de Hausman (Efectos Fijos vs Aleatorios)

**Funciones principales:**
- `verificar_varianza()`: Detecta variables con varianza nula o NAs excesivos
- `validar_entidades_panel2()`: Identifica entidades con varianza nula
- `test_hausman_panel()`: Compara modelos FE vs RE

**Salidas:**
- `diagnosticos/verificacion_varianza_panel1.xlsx`
- `diagnosticos/verificacion_varianza_panel2.xlsx`
- `diagnosticos/validacion_entidades_panel2.xlsx`
- `diagnosticos/hausman_panel1.txt`
- `diagnosticos/hausman_panel2.txt`

**Criterios de validación:**
- ✅ Varianza > 0.001
- ✅ NAs < 5%
- ✅ Test Hausman rechaza H0 (p < 0.05) → usar FE

---

### ✅ Módulo Fase 3.2: Estimación de Modelos Finales
**Archivo:** `modulo_fase32_estimacion.py`

**Objetivos:**
1. Estimar Panel 1 (Aportes) - Modelo RESTRINGIDO
2. Estimar Panel 2 (Reasignación) - Modelo RESTRINGIDO + Desagregado
3. Guardar resultados en formato publicable

**Funciones principales:**
- `estimar_panel_restringido()`: Estima modelo con efectos fijos
- `estimar_panel_por_grupo()`: Estima modelos separados por grupo
- `crear_tabla_comparativa()`: Genera tabla estilo Stargazer

**Modelos estimados:**

#### Panel 1: Aportes (RESTRINGIDO)
```
ln(Aportes_AFP) ~ PC1_Global_c + PC1_Sistematico_c + D_COVID + 
                  Int_Global_COVID + Int_Sistematico_COVID +
                  Controles_centrados + Dummies_Mes + 
                  FE_AFP
```

#### Panel 2: Reasignación
- **Agregado:** Todas las entidades (AFP × Fondo)
- **Desagregado:** Por cada TipodeFondo (0, 1, 2, 3)

**Salidas:**
- `tablas/modelo_panel_1_aportes.txt`
- `tablas/coeficientes_panel_1_aportes.xlsx`
- `tablas/modelo_panel_2_reasignacion_agregado.txt`
- `tablas/modelo_panel_2_reasignacion_tipodefondon_N.txt` (para cada fondo)
- `tablas/tabla_comparativa_panel2.xlsx`

---

### ⏳ Módulo Fase 3.3: Validación de Supuestos (PRÓXIMO)
**Archivo:** `modulo_fase33_supuestos.py`

**Pruebas a implementar:**
1. **Normalidad de residuos** (Test Jarque-Bera + Q-Q plots)
2. **Autocorrelación** (Test de Wooldridge para panel)
3. **Heterocedasticidad** (Test de Breusch-Pagan)
4. **Multicolinealidad** (VIF entre predictores)

**Salidas esperadas:**
- `diagnosticos/normalidad_panel1.png` (Q-Q plot)
- `diagnosticos/autocorrelacion_panel1.txt`
- `diagnosticos/heterocedasticidad_panel1.txt`

---

### ⏳ Módulo Fase 3.4: Análisis de Robustez (PRÓXIMO)
**Archivo:** `modulo_fase34_robustez.py`

**5 Análisis de robustez:**
1. **Sin período COVID** (datos hasta 2020-02)
2. **PC1_Global solo** y **PC1_Sistemático solo**
3. **Ventanas COVID alternativas** (2020-02, 2020-04, 2020-06)
4. **Sin trimestres extremos** (excluir Q1 2020)
5. **Modelo dinámico** (incluir VD rezagada en Panel 1)

**Salidas esperadas:**
- `robustez/panel1_sin_covid.xlsx`
- `robustez/panel1_solo_global.xlsx`
- `robustez/comparacion_robustez.xlsx`

---

### ⏳ Módulo Fase 4: Test de Hipótesis (PRÓXIMO)
**Archivo:** `modulo_fase4_hipotesis.py`

**Hipótesis a testear:**

#### H1 (Panel 1 - Aportes):
- H1.1: β₁(PC1_Global_c) < 0 y p < 0.05
- H1.2: β₂(PC1_Sistematico_c) ≠ 0 y p < 0.10
- H1.3: β₆(Int_Global_COVID) ≠ 0

#### H2 (Panel 2 - Flight-to-Quality):
- H2.1: β₁(Fondo0) > β₁(Fondo3)
- H2.2: Todos los fondos: β₁ < 0

#### H3 (Panel 3 - Composición):
- H3.1: |β₁(Minería)| > |β₁(Soberano)|
- H3.2: |β₂(Local)| > |β₂(Extranjero)|

**Salidas esperadas:**
- `tablas/test_hipotesis_resultados.xlsx`
- `tablas/tabla_hipotesis_vs_resultados.xlsx`

---

### ⏳ Módulo Fase 5: Documentación Final (PRÓXIMO)
**Archivo:** `modulo_fase5_documentacion.py`

**Elementos a generar:**
1. **Tablas de publicación** (estilo Stargazer/APA)
2. **Gráficos de defensa:**
   - Forest plot (coeficientes Panel 2 por Fondo)
   - Heatmap (presión liquidez Panel 3)
3. **Codebook** de variables
4. **Reporte ejecutivo** consolidado

---

## 🚀 INSTRUCCIONES DE USO <a name="instrucciones"></a>

### OPCIÓN 1: Ejecutar Todo Automáticamente (RECOMENDADO)

```bash
# 1. Asegúrate de estar en el directorio correcto
cd /ruta/a/tu/proyecto

# 2. Verifica que tengas todos los archivos
ls -la

# 3. Ejecuta el notebook maestro
python notebook_maestro.py
```

**Ventajas:**
- ✅ Ejecuta todos los módulos secuencialmente
- ✅ Genera reporte consolidado automáticamente
- ✅ Captura y reporta errores

---

### OPCIÓN 2: Ejecutar en Jupyter Notebook

1. **Crear nuevo notebook:**
   ```
   jupyter notebook
   ```

2. **Celda 1 - Configuración:**
   ```python
   # Copiar y pegar TODO el contenido de modulo_0_config.py
   ```

3. **Celda 2 - Verificación:**
   ```python
   # Copiar y pegar contenido de modulo_fase31_verificacion.py
   ```

4. **Celda 3 - Estimación:**
   ```python
   # Copiar y pegar contenido de modulo_fase32_estimacion.py
   ```

5. **Ejecutar celdas secuencialmente** (Shift + Enter)

---

### OPCIÓN 3: Ejecutar Módulos Individuales

Si prefieres control total:

```bash
# Paso 1: Configuración
python modulo_0_config.py

# Paso 2: Verificación Pre-Estimación
python modulo_fase31_verificacion.py

# Paso 3: Estimación de Modelos
python modulo_fase32_estimacion.py

# (Próximos pasos cuando estén disponibles)
# python modulo_fase33_supuestos.py
# python modulo_fase34_robustez.py
# python modulo_fase4_hipotesis.py
# python modulo_fase5_documentacion.py
```

---

## 📊 RESULTADOS ESPERADOS <a name="resultados"></a>

### Estructura de Carpeta de Salida

```
resultados_tesis_20251031_143022/
│
├── 📁 diagnosticos/
│   ├── verificacion_varianza_panel1.xlsx      [Fase 3.1]
│   ├── verificacion_varianza_panel2.xlsx      [Fase 3.1]
│   ├── validacion_entidades_panel2.xlsx       [Fase 3.1]
│   ├── hausman_panel1.txt                     [Fase 3.1]
│   ├── hausman_panel2.txt                     [Fase 3.1]
│   ├── normalidad_panel1.png                  [Fase 3.3]
│   ├── normalidad_panel2.png                  [Fase 3.3]
│   ├── autocorrelacion_panel1.txt             [Fase 3.3]
│   └── autocorrelacion_panel2.txt             [Fase 3.3]
│
├── 📁 tablas/
│   ├── modelo_panel_1_aportes.txt             [Fase 3.2]
│   ├── coeficientes_panel_1_aportes.xlsx      [Fase 3.2]
│   ├── modelo_panel_2_reasignacion_agregado.txt  [Fase 3.2]
│   ├── tabla_comparativa_panel2.xlsx          [Fase 3.2]
│   ├── test_hipotesis_resultados.xlsx         [Fase 4]
│   └── tabla_publicacion_final.xlsx           [Fase 5]
│
├── 📁 graficos/
│   ├── forest_plot_panel2_fondos.png          [Fase 5]
│   ├── heatmap_presion_liquidez_panel3.png    [Fase 5]
│   └── comparacion_robustez.png               [Fase 3.4]
│
├── 📁 robustez/
│   ├── panel1_sin_covid.xlsx                  [Fase 3.4]
│   ├── panel1_solo_global.xlsx                [Fase 3.4]
│   ├── panel1_solo_sistematico.xlsx           [Fase 3.4]
│   ├── panel1_ventanas_alternativas.xlsx      [Fase 3.4]
│   └── comparacion_robustez.xlsx              [Fase 3.4]
│
├── 📁 panel3/
│   ├── ecuacion_F1_Local_Mineria/
│   ├── ecuacion_F1_Local_Soberano/
│   └── ... (41 carpetas, una por ecuación)
│
└── 📄 REPORTE_FINAL.txt                       [Auto-generado]
```

---

## ❗ SOLUCIÓN DE PROBLEMAS <a name="problemas"></a>

### Error: "Archivo no encontrado"

**Causa:** Los archivos Excel no están en el directorio correcto.

**Solución:**
```python
import os
print("Directorio actual:", os.getcwd())
print("Archivos disponibles:", os.listdir())

# Si necesitas cambiar de directorio:
os.chdir('/ruta/a/tus/archivos')
```

---

### Error: "ModuleNotFoundError: No module named 'linearmodels'"

**Causa:** Librerías no instaladas.

**Solución:**
```bash
pip install pandas numpy openpyxl
pip install linearmodels statsmodels scipy
pip install matplotlib seaborn
```

---

### Error: "KeyError: 'Columna_X'"

**Causa:** Nombre de columna incorrecto en Excel.

**Solución:**
```python
# Verificar nombres de columnas
import pandas as pd
df = pd.read_excel('panel_1_aportes.xlsx')
print(df.columns.tolist())

# Ajustar CONFIG si es necesario
CONFIG.VAR_Y_PANEL1 = 'nombre_correcto_columna'
```

---

### Advertencia: "Varianza nula" en verificación

**Causa:** Variable constante o sin variación suficiente.

**Interpretación:**
- ✅ **Normal** en Panel 3 (fondos conservadores no invierten en ciertos sectores)
- ⚠️ **Revisar** en Panel 1 y 2 (puede indicar problema de datos)

**Acción:**
- Verificar datos originales
- Excluir variable si es constante por diseño

---

### Panel 2: R² muy bajo (< 0.01)

**Causa:** Decisiones individuales tienen mucho ruido idiosincrático.

**Interpretación:**
- ✅ **Esperado** según validación doctoral
- No invalida análisis, pero limita inferencia causal
- Usar Panel 2 como análisis **exploratorio**, no confirmatorio

---

## 📞 SOPORTE

Si encuentras problemas:

1. **Revisa el reporte de error** completo
2. **Verifica datos** (columnas, tipos, NAs)
3. **Consulta la documentación** de librerías:
   - [linearmodels](https://bashtage.github.io/linearmodels/)
   - [statsmodels](https://www.statsmodels.org/)
4. **Contacta al autor** con detalles del error

---

## ✅ CHECKLIST ANTES DE EJECUTAR

- [ ] Todos los archivos Excel están en el directorio
- [ ] Librerías Python instaladas
- [ ] Nombres de columnas verificados
- [ ] Python 3.8+ instalado
- [ ] Espacio en disco suficiente (>100MB para resultados)

---

## 📚 REFERENCIAS METODOLÓGICAS

- **Sampieri, R. H. (2018).** Metodología de la investigación (7ª ed.).
- **Samaja, J. (1994).** Epistemología y metodología.
- **Kerlinger, F. N. (1986).** Foundations of behavioral research.
- **Tamayo, M. (2003).** El proceso de la investigación científica.

---

## 📅 ROADMAP DE DESARROLLO

| Módulo | Estado | Fecha Estimada |
|--------|--------|----------------|
| 0: Configuración | ✅ Completo | - |
| 3.1: Verificación | ✅ Completo | - |
| 3.2: Estimación | ✅ Completo | - |
| 3.3: Supuestos | ⏳ En desarrollo | Próximo |
| 3.4: Robustez | 📋 Planeado | Próximo |
| 4: Hipótesis | 📋 Planeado | Próximo |
| 5: Documentación | 📋 Planeado | Próximo |

---

¡Éxito en tu análisis doctoral! 🎓
