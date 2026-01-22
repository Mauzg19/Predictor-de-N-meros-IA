from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
from database import *
from metricas import ModelMetrics
import random
import asyncio

# ============ CONFIGURACIÓN ============
TOKEN = "TU_TOKEN_AQUI"  # Obtén tu token de @BotFather
CHAT_ID = "TU_CHAT_ID"   # Tu ID de chat

# Cargar modelo
try:
    model = load_model("modelo_lstm.keras", compile=False)
    scaler = pickle.load(open("scaler.pkl", "rb"))
    df = pd.read_csv("numeros.csv")
    metrics = ModelMetrics()
    print("✓ Modelo cargado para Bot")
except Exception as e:
    print(f"Error: {e}")

inicializar_bd()

# ============ GENERADOR DE PREDICCIONES ============
ultima_entrada_global = None

def obtener_prediccion_bot():
    """Generar predicción para el bot"""
    global ultima_entrada_global
    
    try:
        numeros = df["numero"].astype(str)
        data = []
        for n in numeros:
            data.extend([int(d) for d in n])
        
        data = np.array(data).reshape(-1, 1)
        data = scaler.transform(data)
        
        inicio_aleatorio = max(0, len(data) - random.randint(5, 20))
        entrada_inicial = data[inicio_aleatorio:inicio_aleatorio + 5]
        
        if len(entrada_inicial) < 5:
            entrada_inicial = data[-5:]
        
        entrada_temporal = entrada_inicial.reshape(1, 5, 1)
        predicciones = []
        
        for i in range(4):
            pred = model.predict(entrada_temporal, verbose=0)
            pred_valor = scaler.inverse_transform(pred)[0][0]
            ruido = np.random.normal(0, 0.5)
            pred_valor_ruidoso = pred_valor + ruido
            pred_valor_ruidoso = np.clip(pred_valor_ruidoso, 0, 9)
            digito = int(np.round(pred_valor_ruidoso))
            digito = np.clip(digito, 0, 9)
            predicciones.append(digito)
            
            digito_escalado = scaler.transform(np.array([[digito]]))[0][0]
            entrada_temporal = np.append(entrada_temporal[:, 1:, :], 
                                        np.array([[[digito_escalado]]]), axis=1)
        
        ultima_entrada_global = entrada_temporal.copy()
        return ''.join(map(str, predicciones))
    except Exception as e:
        return f"Error: {e}"

# ============ MANEJADORES DE COMANDOS ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    mensaje = """
🎰 *Predictor de Números IA* 🎰

¡Bienvenido! Este bot predice números usando una Red Neuronal LSTM.

*Comandos disponibles:*
/prediccion - Obtener 4 números predichos
/estadisticas - Ver estadísticas del modelo
/historial - Ver últimas 10 predicciones
/help - Ver esta ayuda

*Nota:* Las predicciones son basadas en análisis de datos históricos.
    """
    await update.message.reply_text(mensaje, parse_mode='Markdown')

async def prediccion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /prediccion"""
    try:
        numeros = obtener_prediccion_bot()
        confianza = metrics.calcular_confianza_general()
        
        guardar_prediccion(numeros, confianza)
        
        mensaje = f"""
🔮 *Predicción del Día*

Números predichos: `{numeros}`
Confianza: {confianza}%
Hora: {datetime.now().strftime('%H:%M:%S')}

*Recuerda:* Estas son predicciones basadas en patrones históricos.
        """
        await update.message.reply_text(mensaje, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def estadisticas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /estadisticas"""
    try:
        stats = obtener_estadisticas_generales()
        
        mensaje = f"""
📊 *Estadísticas del Modelo*

Total de predicciones: {stats.get('total_predicciones', 0)}
Comparaciones realizadas: {stats.get('total_comparaciones', 0)}
Tasa promedio de acierto: {stats.get('tasa_promedio_acierto', 0)}%
Secuencias acertadas: {stats.get('secuencias_acertadas', 0)}
Confianza general: {metrics.calcular_confianza_general()}%

Mejor predicción: {stats.get('mejor_prediccion', '-')} ({stats.get('mejor_porcentaje', 0)}%)
Peor predicción: {stats.get('peor_prediccion', '-')} ({stats.get('peor_porcentaje', 0)}%)
        """
        await update.message.reply_text(mensaje, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def historial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /historial"""
    try:
        predicciones = obtener_ultimas_predicciones(10)
        
        mensaje = "*📜 Últimas 10 Predicciones*\n\n"
        
        for i, pred in enumerate(predicciones, 1):
            fecha = pred['fecha'].split(' ')[0] if ' ' in pred['fecha'] else pred['fecha']
            confianza = pred['confianza'] if pred['confianza'] else 'N/A'
            mensaje += f"{i}. `{pred['numeros']}` - Conf: {confianza}% ({fecha})\n"
        
        await update.message.reply_text(mensaje, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /help"""
    await start(update, context)

async def manejo_general(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manejo de mensajes generales"""
    await update.message.reply_text(
        "No entiendo ese comando. Usa /help para ver los comandos disponibles.",
        parse_mode='Markdown'
    )

# ============ ENVÍO AUTOMÁTICO DIARIO ============
async def enviar_prediccion_diaria(context: ContextTypes.DEFAULT_TYPE):
    """Enviar predicción automática a las 9:00 AM"""
    try:
        numeros = obtener_prediccion_bot()
        confianza = metrics.calcular_confianza_general()
        
        guardar_prediccion(numeros, confianza)
        
        mensaje = f"""
🎰 *Predicción Diaria* 🎰

Números del día: `{numeros}`
Confianza: {confianza}%
Hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}

¡Buena suerte! 🍀
        """
        
        bot = Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=mensaje, parse_mode='Markdown')
        print(f"✓ Predicción diaria enviada: {numeros}")
    except Exception as e:
        print(f"✗ Error enviando predicción diaria: {e}")

# ============ INICIAR BOT ============
def main():
    """Iniciar el bot"""
    print("🤖 Iniciando Bot de Telegram...")
    
    if TOKEN == "TU_TOKEN_AQUI" or CHAT_ID == "TU_CHAT_ID":
        print("⚠️  CONFIGURAR: Edita TOKEN y CHAT_ID en bot_telegram.py")
        print("   Obtén token en: https://t.me/BotFather")
        return
    
    app = Application.builder().token(TOKEN).build()
    
    # Manejadores de comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("prediccion", prediccion))
    app.add_handler(CommandHandler("estadisticas", estadisticas))
    app.add_handler(CommandHandler("historial", historial))
    app.add_handler(CommandHandler("help", help_command))
    
    # Manejador de mensajes generales
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejo_general))
    
    # Predicción automática diaria (9:00 AM)
    # Descomentar para activar
    # job_queue = app.job_queue
    # job_queue.run_daily(enviar_prediccion_diaria, time=datetime.time(hour=9, minute=0))
    
    print("✓ Bot iniciado. Presiona Ctrl+C para detener.")
    app.run_polling()

if __name__ == "__main__":
    main()

