import sqlite3
from datetime import datetime
import os

# ============ CONFIGURACIÓN DE BD ============
DB_PATH = "predicciones.db"

def inicializar_bd():
    """Crear tablas de la base de datos"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabla de predicciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            numeros_predichos TEXT NOT NULL,
            confianza REAL,
            punto_inicio TEXT,
            UNIQUE(fecha, numeros_predichos)
        )
    ''')
    
    # Tabla de resultados reales (ganadores)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_reales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha DATE UNIQUE NOT NULL,
            numeros_ganadores TEXT NOT NULL
        )
    ''')
    
    # Tabla de comparaciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediccion_id INTEGER,
            resultado_id INTEGER,
            aciertos_totales INTEGER,
            aciertos_secuencia INTEGER,
            porcentaje_acierto REAL,
            FOREIGN KEY(prediccion_id) REFERENCES predicciones(id),
            FOREIGN KEY(resultado_id) REFERENCES resultados_reales(id)
        )
    ''')
    
    # Tabla de estadísticas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS estadisticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_predicciones INTEGER,
            aciertos INTEGER,
            tasa_acierto_promedio REAL,
            mejor_prediccion TEXT,
            peor_prediccion TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Base de datos inicializada")

# ============ FUNCIONES PARA PREDICCIONES ============
def guardar_prediccion(numeros, confianza=None, punto_inicio=None):
    """Guardar una predicción en la BD"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predicciones (numeros_predichos, confianza, punto_inicio)
            VALUES (?, ?, ?)
        ''', (str(numeros), confianza, punto_inicio))
        
        conn.commit()
        prediccion_id = cursor.lastrowid
        conn.close()
        
        print(f"✓ Predicción guardada: {numeros} (ID: {prediccion_id})")
        return prediccion_id
    except sqlite3.IntegrityError:
        print("⚠ Predicción duplicada, no guardada")
        return None
    except Exception as e:
        print(f"✗ Error al guardar predicción: {e}")
        return None

def obtener_ultimas_predicciones(limite=10):
    """Obtener las últimas predicciones"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, fecha, numeros_predichos, confianza
            FROM predicciones
            ORDER BY fecha DESC
            LIMIT ?
        ''', (limite,))
        
        resultados = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": r[0],
                "fecha": r[1],
                "numeros": r[2],
                "confianza": r[3]
            }
            for r in resultados
        ]
    except Exception as e:
        print(f"✗ Error al obtener predicciones: {e}")
        return []

# ============ FUNCIONES PARA RESULTADOS REALES ============
def guardar_resultado_real(fecha, numeros_ganadores):
    """Guardar números ganadores del día"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resultados_reales (fecha, numeros_ganadores)
            VALUES (?, ?)
        ''', (fecha, str(numeros_ganadores)))
        
        conn.commit()
        resultado_id = cursor.lastrowid
        conn.close()
        
        print(f"✓ Resultado real guardado: {numeros_ganadores} (ID: {resultado_id})")
        return resultado_id
    except Exception as e:
        print(f"✗ Error al guardar resultado: {e}")
        return None

def obtener_resultado_por_fecha(fecha):
    """Obtener el resultado ganador de una fecha"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, numeros_ganadores
            FROM resultados_reales
            WHERE fecha = ?
        ''', (fecha,))
        
        resultado = cursor.fetchone()
        conn.close()
        
        if resultado:
            return {"id": resultado[0], "numeros": resultado[1]}
        return None
    except Exception as e:
        print(f"✗ Error al obtener resultado: {e}")
        return None

# ============ FUNCIONES DE COMPARACIÓN ============
def comparar_prediccion_con_resultado(prediccion_id, resultado_id):
    """Comparar predicción con resultado real"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT numeros_predichos FROM predicciones WHERE id = ?
        ''', (prediccion_id,))
        prediccion = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT numeros_ganadores FROM resultados_reales WHERE id = ?
        ''', (resultado_id,))
        resultado = cursor.fetchone()[0]
        
        # Contar aciertos
        aciertos_totales = sum(1 for p, r in zip(str(prediccion), str(resultado)) if p == r)
        aciertos_secuencia = 1 if str(prediccion) == str(resultado) else 0
        porcentaje_acierto = (aciertos_totales / 4) * 100
        
        # Guardar comparación
        cursor.execute('''
            INSERT INTO comparaciones (prediccion_id, resultado_id, aciertos_totales, 
                                      aciertos_secuencia, porcentaje_acierto)
            VALUES (?, ?, ?, ?, ?)
        ''', (prediccion_id, resultado_id, aciertos_totales, aciertos_secuencia, porcentaje_acierto))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Comparación guardada: {aciertos_totales}/4 aciertos ({porcentaje_acierto:.1f}%)")
        
        return {
            "aciertos_totales": aciertos_totales,
            "aciertos_secuencia": aciertos_secuencia,
            "porcentaje_acierto": porcentaje_acierto
        }
    except Exception as e:
        print(f"✗ Error en comparación: {e}")
        return None

# ============ FUNCIONES DE ESTADÍSTICAS ============
def obtener_estadisticas_generales():
    """Obtener estadísticas generales de predicciones"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total de predicciones
        cursor.execute('SELECT COUNT(*) FROM predicciones')
        total_predicciones = cursor.fetchone()[0]
        
        # Comparaciones realizadas
        cursor.execute('SELECT COUNT(*) FROM comparaciones')
        total_comparaciones = cursor.fetchone()[0]
        
        if total_comparaciones > 0:
            # Tasa promedio de acierto
            cursor.execute('SELECT AVG(porcentaje_acierto) FROM comparaciones')
            tasa_promedio = cursor.fetchone()[0]
            
            # Secuencias acertadas
            cursor.execute('SELECT SUM(aciertos_secuencia) FROM comparaciones')
            aciertos_secuencia_total = cursor.fetchone()[0] or 0
            
            # Mejor predicción
            cursor.execute('''
                SELECT p.numeros_predichos, MAX(c.porcentaje_acierto)
                FROM comparaciones c
                JOIN predicciones p ON c.prediccion_id = p.id
            ''')
            mejor = cursor.fetchone()
            
            # Peor predicción
            cursor.execute('''
                SELECT p.numeros_predichos, MIN(c.porcentaje_acierto)
                FROM comparaciones c
                JOIN predicciones p ON c.prediccion_id = p.id
            ''')
            peor = cursor.fetchone()
        else:
            tasa_promedio = 0
            aciertos_secuencia_total = 0
            mejor = None
            peor = None
        
        conn.close()
        
        return {
            "total_predicciones": total_predicciones,
            "total_comparaciones": total_comparaciones,
            "tasa_promedio_acierto": round(tasa_promedio, 2) if tasa_promedio else 0,
            "secuencias_acertadas": aciertos_secuencia_total,
            "mejor_prediccion": mejor[0] if mejor else "-",
            "mejor_porcentaje": round(mejor[1], 2) if mejor else 0,
            "peor_prediccion": peor[0] if peor else "-",
            "peor_porcentaje": round(peor[1], 2) if peor else 0
        }
    except Exception as e:
        print(f"✗ Error al obtener estadísticas: {e}")
        return {}

def obtener_historial_comparaciones(limite=10):
    """Obtener historial de comparaciones"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.fecha, p.numeros_predichos, r.numeros_ganadores, 
                   c.aciertos_totales, c.porcentaje_acierto, c.aciertos_secuencia
            FROM comparaciones c
            JOIN predicciones p ON c.prediccion_id = p.id
            JOIN resultados_reales r ON c.resultado_id = r.id
            ORDER BY p.fecha DESC
            LIMIT ?
        ''', (limite,))
        
        resultados = cursor.fetchall()
        conn.close()
        
        return [
            {
                "fecha": r[0],
                "prediccion": r[1],
                "resultado": r[2],
                "aciertos": r[3],
                "porcentaje": r[4],
                "secuencia_completa": bool(r[5])
            }
            for r in resultados
        ]
    except Exception as e:
        print(f"✗ Error al obtener historial: {e}")
        return []

# ============ INICIALIZAR BD ============
if __name__ == "__main__":
    inicializar_bd()
    print("\n✓ Base de datos lista para usar")
