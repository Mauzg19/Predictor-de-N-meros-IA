from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
from collections import Counter
import json
import random
from database import *
from metricas import ModelMetrics

app = Flask(__name__)

# ============ CARGAR MODELO Y DATOS ============
try:
    model = load_model("modelo_lstm.keras", compile=False)
    scaler = pickle.load(open("scaler.pkl", "rb"))
    df = pd.read_csv("numeros.csv")
    metrics = ModelMetrics()
    print("✓ Modelo cargado exitosamente")
    print(f"✓ Confianza general: {metrics.calcular_confianza_general()}%")
except Exception as e:
    print(f"Error cargando modelo: {e}")

# Inicializar BD
inicializar_bd()

# Variable global para almacenar la última predicción
ultima_entrada_global = None
ultima_prediccion = None

# ============ FUNCIONES AUXILIARES ============
def obtener_predicciones():
    """Genera 4 predicciones usando el modelo LSTM con variabilidad"""
    global ultima_entrada_global
    
    try:
        numeros = df["numero"].astype(str)
        data = []
        for n in numeros:
            data.extend([int(d) for d in n])
        
        data = np.array(data).reshape(-1, 1)
        data = scaler.transform(data)
        
        # Seleccionar un punto aleatorio de los últimos 20 dígitos para más variabilidad
        inicio_aleatorio = max(0, len(data) - random.randint(5, 20))
        entrada_inicial = data[inicio_aleatorio:inicio_aleatorio + 5]
        
        # Si no hay suficientes dígitos, usar los últimos 5
        if len(entrada_inicial) < 5:
            entrada_inicial = data[-5:]
        
        entrada_temporal = entrada_inicial.reshape(1, 5, 1)
        
        predicciones = []
        
        for i in range(4):
            # Obtener predicción del modelo
            pred = model.predict(entrada_temporal, verbose=0)
            
            # Aplicar temperature sampling para variabilidad
            # Convertir predicción normalizada a escala original
            pred_valor = scaler.inverse_transform(pred)[0][0]
            
            # Añadir ruido gaussiano pequeño para variabilidad
            ruido = np.random.normal(0, 0.5)
            pred_valor_ruidoso = pred_valor + ruido
            
            # Asegurar que está en rango [0, 9]
            pred_valor_ruidoso = np.clip(pred_valor_ruidoso, 0, 9)
            
            # Redondear al dígito más cercano
            digito = int(np.round(pred_valor_ruidoso))
            digito = np.clip(digito, 0, 9)
            
            predicciones.append(digito)
            
            # Preparar entrada para siguiente predicción
            digito_escalado = scaler.transform(np.array([[digito]]))[0][0]
            entrada_temporal = np.append(entrada_temporal[:, 1:, :], 
                                        np.array([[[digito_escalado]]]), axis=1)
        
        # Guardar la última entrada para referencia
        ultima_entrada_global = entrada_temporal.copy()
        
        return ''.join(map(str, predicciones))
    except Exception as e:
        return f"Error: {e}"

def obtener_analisis():
    """Calcula análisis de patrones"""
    try:
        todos = "".join(df["numero"].astype(str))
        frecuencia = Counter(todos)
        total_digitos = len(todos)
        
        # Frecuencia con porcentajes
        frecuencia_datos = []
        for d in sorted(frecuencia.keys()):
            f = frecuencia[d]
            porcentaje = (f / total_digitos) * 100
            frecuencia_datos.append({
                "digito": d,
                "cantidad": f,
                "porcentaje": round(porcentaje, 2)
            })
        
        # Estadísticas
        digitos_numeros = [int(d) for d in todos]
        estadisticas = {
            "media": round(np.mean(digitos_numeros), 2),
            "mediana": round(np.median(digitos_numeros), 2),
            "desv_estandar": round(np.std(digitos_numeros), 2),
            "minimo": int(np.min(digitos_numeros)),
            "maximo": int(np.max(digitos_numeros)),
            "rango": int(np.max(digitos_numeros) - np.min(digitos_numeros)),
            "total_digitos": total_digitos
        }
        
        # Transiciones
        transiciones = {}
        for i in range(len(todos) - 1):
            par = todos[i] + "->" + todos[i+1]
            transiciones[par] = transiciones.get(par, 0) + 1
        
        transiciones_ordenadas = sorted(transiciones.items(), key=lambda x: x[1], reverse=True)[:10]
        transiciones_datos = [{
            "transicion": k,
            "cantidad": v,
            "porcentaje": round((v / (total_digitos - 1)) * 100, 2)
        } for k, v in transiciones_ordenadas]
        
        return {
            "frecuencia": frecuencia_datos,
            "estadisticas": estadisticas,
            "transiciones": transiciones_datos
        }
    except Exception as e:
        return {"error": str(e)}

# ============ RUTAS ============
@app.route("/")
def home():
    prediccion = obtener_predicciones()
    analisis = obtener_analisis()
    return render_template("index.html", prediccion=prediccion, analisis=analisis)

@app.route("/api/prediccion")
def api_prediccion():
    global ultima_prediccion
    try:
        prediccion = obtener_predicciones()
        confianza = metrics.calcular_confianza_general()
        
        # Calcular confianza individual
        if ultima_prediccion:
            confianza_individual = metrics.obtener_confianza_prediccion(ultima_prediccion, prediccion)
        else:
            confianza_individual = confianza
        
        # Guardar en BD
        guardar_prediccion(prediccion, confianza_individual)
        
        ultima_prediccion = prediccion
        
        print(f"✓ Predicción generada: {prediccion} (Confianza: {confianza_individual}%)")
        response = jsonify({
            "prediccion": str(prediccion),
            "confianza": round(confianza_individual, 2)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"✗ Error en /api/prediccion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analisis")
def api_analisis():
    analisis = obtener_analisis()
    return jsonify(analisis)

@app.route("/api/metricas")
def api_metricas():
    """Endpoint para obtener métricas del modelo"""
    return jsonify(metrics.obtener_reporte_json())

@app.route("/api/historial")
def api_historial():
    """Endpoint para obtener historial de predicciones"""
    limite = request.args.get('limite', 10, type=int)
    historial = obtener_ultimas_predicciones(limite)
    return jsonify({"predicciones": historial})

@app.route("/api/estadisticas-bd")
def api_estadisticas_bd():
    """Endpoint para obtener estadísticas de la BD"""
    stats = obtener_estadisticas_generales()
    return jsonify(stats)

@app.route("/api/resultado-real", methods=['POST'])
def api_resultado_real():
    """Endpoint para guardar resultado real (ganador)"""
    try:
        data = request.json
        fecha = data.get('fecha')
        numeros = data.get('numeros')
        
        if not fecha or not numeros:
            return jsonify({"error": "Faltan datos"}), 400
        
        resultado_id = guardar_resultado_real(fecha, numeros)
        return jsonify({"success": True, "id": resultado_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/comparar", methods=['POST'])
def api_comparar():
    """Endpoint para comparar predicción con resultado"""
    try:
        data = request.json
        prediccion_id = data.get('prediccion_id')
        resultado_id = data.get('resultado_id')
        
        if not prediccion_id or not resultado_id:
            return jsonify({"error": "Faltan datos"}), 400
        
        comparacion = comparar_prediccion_con_resultado(prediccion_id, resultado_id)
        return jsonify(comparacion)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
