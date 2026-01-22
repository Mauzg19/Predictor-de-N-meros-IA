# 🔮 Sistema de Predicción de Números con IA

Sistema avanzado de predicción de números usando Red Neuronal LSTM, con interfaz web, bot de Telegram y análisis detallado de patrones.

## ✨ Características Principales

### 1. **Red Neuronal LSTM Mejorada**
- ✓ 3 capas LSTM con Dropout para regularización
- ✓ Early Stopping para evitar sobreajuste
- ✓ Validación cruzada del modelo
- ✓ Métricas detalladas (precisión, RMSE, MAE)
- ✓ Gráficos de entrenamiento

### 2. **Base de Datos SQLite**
- ✓ Almacenamiento de predicciones
- ✓ Registro de resultados reales
- ✓ Comparación automática predicción vs resultado
- ✓ Historial completo con estadísticas

### 3. **Dashboard Web Avanzado**
- ✓ Interfaz moderna con Bootstrap
- ✓ Tema oscuro/claro
- ✓ Pestañas: Inicio, Análisis, Métricas, Historial
- ✓ Indicador de confianza en predicciones
- ✓ Gráficos interactivos
- ✓ Historial de últimas 10 predicciones
- ✓ Responsive design

### 4. **Bot de Telegram**
- ✓ Comandos: /prediccion, /estadisticas, /historial, /help
- ✓ Predicciones diarias automáticas
- ✓ Notificaciones de resultados
- ✓ Estadísticas en tiempo real

### 5. **API REST Completa**
- `/api/prediccion` - Obtener predicción con confianza
- `/api/analisis` - Análisis de patrones
- `/api/metricas` - Métricas del modelo
- `/api/historial` - Historial de predicciones
- `/api/estadisticas-bd` - Estadísticas generales
- `/api/resultado-real` - Guardar resultado real
- `/api/comparar` - Comparar predicción con resultado

## 📦 Instalación

### 1. Requisitos Previos
- Python 3.8+
- pip

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Archivos Necesarios
- `numeros.csv` - Datos históricos de números

## 🚀 Uso

### A. Entrenar el Modelo
```bash
python entrenar_modelo.py
```

**Genera:**
- `modelo_lstm.keras` - Modelo entrenado
- `scaler.pkl` - Normalizador de datos
- `metricas_modelo.json` - Métricas de entrenamiento
- `grafico_entrenamiento.png` - Gráficos de entrenamiento

### B. Iniciar Aplicación Web
```bash
python app_web.py
```

Abre en navegador: `http://localhost:5000`

**Funcionalidades:**
- Ver predicciones en tiempo real
- Actualizar predicciones con botón
- Ver análisis detallado
- Consultar métricas del modelo
- Revisar historial completo

### C. Iniciar Bot de Telegram
```bash
python bot_telegram.py
```

**Primero configura:**
1. Obtén token de [@BotFather](https://t.me/BotFather)
2. Obtén tu CHAT_ID
3. Edita `bot_telegram.py` y reemplaza:
   - `TOKEN = "TU_TOKEN_AQUI"`
   - `CHAT_ID = "TU_CHAT_ID"`

### D. Ver Análisis de Patrones
```bash
python analisis_patrones.py
```

**Genera:**
- `analisis_frecuencia.csv` - Frecuencias de dígitos
- `analisis_transiciones.csv` - Transiciones entre dígitos
- `analisis_patrones.png` - Gráficos visuales

### E. Ver Métricas Detalladas
```bash
python metricas.py
```

## 📊 Estructura de Archivos

```
prediccion_numeros_IA/
├── entrenar_modelo.py          # Entrenamiento mejorado
├── app_web.py                  # Aplicación Flask
├── bot_telegram.py             # Bot de Telegram
├── analisis_patrones.py        # Análisis de datos
├── metricas.py                 # Sistemas de métricas
├── database.py                 # Gestión de BD SQLite
├── prediccion.py               # Predicción simple
├── numeros.csv                 # Datos históricos
├── requirements.txt            # Dependencias
├── modelo_lstm.keras           # Modelo entrenado
├── scaler.pkl                  # Normalizador
├── metricas_modelo.json        # Métricas guardadas
├── predicciones.db             # Base de datos
└── templates/
    └── index.html              # Interfaz web
```

## 🎯 Mejoras Implementadas

### 1. **Entrenamiento Mejorado**
- Más epochs (100 vs 30)
- Validation split (80/20)
- Early Stopping automático
- Capas Dropout para regularización
- 3 capas LSTM vs 2

### 2. **Métricas de Precisión**
- Exactitud en entrenamiento y validación
- RMSE y MAE
- Validación cruzada
- Gráficos de pérdida
- Confianza calculada automáticamente

### 3. **Base de Datos Completa**
- Tabla de predicciones
- Tabla de resultados reales
- Tabla de comparaciones
- Tabla de estadísticas
- Relaciones automáticas

### 4. **Interfaz Moderna**
- 4 pestañas principales
- Tema oscuro/claro
- Indicador visual de confianza
- Historial interactivo
- Responsive design
- Animaciones suaves

### 5. **Bot Funcional**
- 4 comandos principales
- Predicciones diarias (configurable)
- Estadísticas en tiempo real
- Notificaciones automáticas

### 6. **API Completa**
- 7 endpoints REST
- Gestión de confianza
- Comparación de predicciones
- Estadísticas en BD

## 📈 Cómo Funciona

### Flujo de Predicción:
1. **Carga modelo** LSTM entrenado
2. **Selecciona punto aleatorio** de los últimos datos
3. **Genera 4 dígitos** secuencialmente
4. **Añade ruido gaussiano** para variabilidad
5. **Calcula confianza** basada en exactitud del modelo
6. **Guarda en BD** automáticamente

### Flujo de Comparación:
1. Usuario ingresa números ganadores
2. Sistema compara con predicción
3. Cuenta aciertos (0-4)
4. Calcula porcentaje de acierto
5. Guarda estadísticas en BD
6. Actualiza tasa de acierto general

## 🔧 Configuración Personalizada

### Cambiar Rango de Ruido
En `app_web.py` o `bot_telegram.py`:
```python
ruido = np.random.normal(0, 0.5)  # Cambiar segundo parámetro
```

### Cambiar Número de Predicciones
```python
for i in range(4):  # Cambiar a 3, 5, etc.
```

### Cambiar Modelo de Predicción
En `entrenar_modelo.py`:
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(5, 1)),  # Aumentar neuronas
    Dropout(0.3),  # Aumentar dropout
    # ...
])
```

## 📊 Interpretación de Métricas

- **Exactitud**: % de predicciones exactas
- **RMSE**: Error cuadrático medio (menor es mejor)
- **MAE**: Error absoluto medio (menor es mejor)
- **Confianza**: 0-100% basada en exactitud validación
- **Tasa de Acierto**: % de predicciones acertadas vs resultados reales

## 🐛 Troubleshooting

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: "Port 5000 already in use"
```bash
python app_web.py --port 5001
```

### Error: "Bad token for bot"
Verifica que configuraste correctamente TOKEN y CHAT_ID en `bot_telegram.py`

## 📝 Notas Importantes

- Las predicciones son basadas en **patrones históricos**
- No garantizan resultados reales
- Usar solo para **análisis educativo**
- Mantener datos históricos actualizados
- Entrenar el modelo regularmente con nuevos datos

## 🎓 Próximas Mejoras

- [ ] Validación con APIs externas de resultados
- [ ] Dashboard de reportes PDF
- [ ] Integración con múltiples loterías
- [ ] Modelo con atención (Transformer)
- [ ] Predicción con confianza por dígito
- [ ] WebSocket para actualizaciones en tiempo real

## 📄 Licencia

Este proyecto es de código abierto para uso educativo.

## ✉️ Soporte

Para reportar errores o sugerencias, contacta al desarrollador.

---

**Versión**: 2.0  
**Última actualización**: 22 de enero de 2026  
**Estado**: ✅ Operativo
