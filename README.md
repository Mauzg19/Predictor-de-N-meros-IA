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


