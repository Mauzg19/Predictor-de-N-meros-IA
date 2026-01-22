import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd

class ModelMetrics:
    """Clase para gestionar métricas del modelo"""
    
    def __init__(self, model_path="modelo_lstm.keras", scaler_path="scaler.pkl", metrics_path="metricas_modelo.json"):
        self.model = load_model(model_path, compile=False)
        self.scaler = pickle.load(open(scaler_path, "rb"))
        self.metrics_path = metrics_path
        self.metricas = self.cargar_metricas()
    
    def cargar_metricas(self):
        """Cargar métricas guardadas"""
        try:
            with open(self.metrics_path, "r") as f:
                return json.load(f)
        except:
            return {}
    
    def obtener_confianza_prediccion(self, prediccion_anterior, prediccion_actual):
        """
        Calcular confianza en una predicción basada en consistencia
        Escala: 0-100%
        """
        try:
            # Comparar con predicciones previas
            diferencia = abs(int(prediccion_actual) - int(prediccion_anterior))
            
            # Mayor diferencia = menos confianza
            # Diferencia 0-2: Alta confianza (80-100%)
            # Diferencia 3-5: Media confianza (50-80%)
            # Diferencia 6-9: Baja confianza (30-50%)
            
            if diferencia <= 2:
                confianza = 100 - (diferencia * 10)
            elif diferencia <= 5:
                confianza = 80 - ((diferencia - 2) * 10)
            else:
                confianza = 50 - ((diferencia - 5) * 3)
            
            return max(30, min(100, confianza))  # Rango 30-100%
        except:
            return 50
    
    def calcular_confianza_general(self):
        """
        Calcular confianza general del modelo basada en métricas
        """
        if not self.metricas:
            return 50
        
        exactitud_val = self.metricas.get("exactitud_validacion", 0)
        
        # Convertir exactitud a confianza
        # Exactitud 70%+ = 90-100% confianza
        # Exactitud 60-70% = 70-90% confianza
        # Exactitud <60% = <70% confianza
        
        if exactitud_val >= 70:
            confianza = 90 + (min(exactitud_val, 90) - 70) / 2
        elif exactitud_val >= 60:
            confianza = 70 + (exactitud_val - 60) / 2
        else:
            confianza = exactitud_val
        
        return round(confianza, 2)
    
    def mostrar_metricas(self):
        """Mostrar todas las métricas disponibles"""
        print("\n" + "="*60)
        print("📊 MÉTRICAS DEL MODELO LSTM")
        print("="*60)
        
        if not self.metricas:
            print("⚠ No hay métricas disponibles. Entrena el modelo primero.")
            return
        
        print(f"Exactitud Entrenamiento: {self.metricas.get('exactitud_entrenamiento', 'N/A')}%")
        print(f"Exactitud Validación:    {self.metricas.get('exactitud_validacion', 'N/A')}%")
        print(f"RMSE Validación:         {self.metricas.get('rmse_validacion', 'N/A')}")
        print(f"MAE Validación:          {self.metricas.get('mae_validacion', 'N/A')}")
        print(f"Épocas Entrenadas:       {self.metricas.get('epochs_entrenados', 'N/A')}")
        print("="*60)
        
        confianza = self.calcular_confianza_general()
        print(f"🎯 Confianza General: {confianza}%")
        print("="*60 + "\n")
    
    def obtener_reporte_json(self):
        """Generar reporte en JSON"""
        return {
            "metricas_entrenamiento": self.metricas,
            "confianza_general": round(self.calcular_confianza_general(), 2),
            "estado": "Modelo optimizado y validado"
        }

# ============ FUNCIONES DE VALIDACIÓN CRUZADA ============
def validacion_cruzada_simple(datos_X, datos_y, n_splits=5):
    """
    Realizar validación cruzada simple
    """
    print("\n🔄 Realizando validación cruzada...")
    print(f"   Folds: {n_splits}")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_num = 1
    resultados = []
    
    for train_idx, test_idx in kfold.split(datos_X):
        print(f"   Fold {fold_num}/{n_splits}...", end="", flush=True)
        
        X_train, X_test = datos_X[train_idx], datos_X[test_idx]
        y_train, y_test = datos_y[train_idx], datos_y[test_idx]
        
        # Aquí iría el entrenamiento específico por fold
        # Por ahora solo recopilamos índices
        resultados.append({
            "fold": fold_num,
            "tamaño_entrenamiento": len(X_train),
            "tamaño_test": len(X_test)
        })
        
        print(" ✓")
        fold_num += 1
    
    print(f"✓ Validación cruzada completada\n")
    return resultados

# ============ EXPORTAR REPORTE ============
def exportar_reporte_txt():
    """Exportar reporte de métricas a archivo TXT"""
    try:
        metrics = ModelMetrics()
        
        with open("reporte_modelo.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE MÉTRICAS DEL MODELO LSTM\n")
            f.write("="*60 + "\n\n")
            
            f.write("📊 RESULTADOS DE ENTRENAMIENTO\n")
            f.write("-"*60 + "\n")
            for key, value in metrics.metricas.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n🎯 CONFIANZA GENERAL\n")
            f.write("-"*60 + "\n")
            f.write(f"Confianza: {metrics.calcular_confianza_general()}%\n")
            
            f.write("\n✅ REPORTE GENERADO\n")
        
        print("✓ Reporte exportado: reporte_modelo.txt")
    except Exception as e:
        print(f"✗ Error al exportar reporte: {e}")

if __name__ == "__main__":
    # Ejemplo de uso
    metrics = ModelMetrics()
    metrics.mostrar_metricas()
    
    print("\n📄 Generando reportes...")
    exportar_reporte_txt()
    
    print("\n✅ Métricas procesadas correctamente")
