import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json

# ============ CARGAR Y PREPARAR DATOS ============
df = pd.read_csv("numeros.csv")
numeros = df["numero"].astype(str)

data = []
for n in numeros:
    data.extend([int(d) for d in n])

data = np.array(data).reshape(-1, 1)

# Normalizar datos
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Preparar secuencias
X, y = [], []
for i in range(len(data_scaled) - 5):
    X.append(data_scaled[i:i+5])
    y.append(data_scaled[i+5])

X, y = np.array(X), np.array(y)

# Dividir en entrenamiento y validación
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"✓ Datos cargados: {len(X)} secuencias")
print(f"  - Entrenamiento: {len(X_train)}")
print(f"  - Validación: {len(X_val)}")

# ============ CONSTRUIR MODELO MEJORADO ============
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(5, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("\n📊 Modelo configurado:")
model.summary()

# ============ EARLY STOPPING ============
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ============ ENTRENAR MODELO ============
print("\n🚀 Entrenando modelo...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ============ EVALUAR MODELO ============
print("\n📈 Evaluando modelo...")
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)

# Predicciones para calcular métricas adicionales
y_pred_train = model.predict(X_train, verbose=0)
y_pred_val = model.predict(X_val, verbose=0)

# Invertir escala para evaluar en rango [0,9]
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))
y_pred_train_original = scaler.inverse_transform(y_pred_train)
y_pred_val_original = scaler.inverse_transform(y_pred_val)

# Redondear predicciones a dígitos enteros
y_pred_train_rounded = np.round(y_pred_train_original).clip(0, 9).astype(int)
y_pred_val_rounded = np.round(y_pred_val_original).clip(0, 9).astype(int)

# Calcular exactitud
train_accuracy = np.mean(y_pred_train_rounded == y_train_original.astype(int)) * 100
val_accuracy = np.mean(y_pred_val_rounded == y_val_original.astype(int)) * 100

rmse_train = np.sqrt(mean_squared_error(y_train_original, y_pred_train_original))
rmse_val = np.sqrt(mean_squared_error(y_val_original, y_pred_val_original))
mae_train = mean_absolute_error(y_train_original, y_pred_train_original)
mae_val = mean_absolute_error(y_val_original, y_pred_val_original)

# ============ MOSTRAR MÉTRICAS ============
print("\n" + "="*60)
print("📊 MÉTRICAS DE RENDIMIENTO")
print("="*60)
print(f"Exactitud Entrenamiento: {train_accuracy:.2f}%")
print(f"Exactitud Validación:    {val_accuracy:.2f}%")
print(f"RMSE Entrenamiento:      {rmse_train:.4f}")
print(f"RMSE Validación:         {rmse_val:.4f}")
print(f"MAE Entrenamiento:       {mae_train:.4f}")
print(f"MAE Validación:          {mae_val:.4f}")
print("="*60)

# ============ GUARDAR MÉTRICAS ============
metricas = {
    "exactitud_entrenamiento": float(train_accuracy),
    "exactitud_validacion": float(val_accuracy),
    "rmse_entrenamiento": float(rmse_train),
    "rmse_validacion": float(rmse_val),
    "mae_entrenamiento": float(mae_train),
    "mae_validacion": float(mae_val),
    "epochs_entrenados": len(history.history['loss'])
}

with open("metricas_modelo.json", "w") as f:
    json.dump(metricas, f, indent=4)

print("\n✓ Métricas guardadas en: metricas_modelo.json")

# ============ GUARDAR MODELO Y SCALER ============
model.save("modelo_lstm.keras")
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("✓ Modelo guardado en: modelo_lstm.keras")
print("✓ Scaler guardado en: scaler.pkl")

# ============ GRÁFICOS DE ENTRENAMIENTO ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Entrenamiento del Modelo LSTM", fontsize=16, fontweight='bold')

# Gráfico 1: Pérdida
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Pérdida Entrenamiento', linewidth=2)
ax1.plot(history.history['val_loss'], label='Pérdida Validación', linewidth=2)
ax1.set_xlabel('Época')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Pérdida del Modelo')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: MAE
ax2 = axes[0, 1]
ax2.plot(history.history['mae'], label='MAE Entrenamiento', linewidth=2)
ax2.plot(history.history['val_mae'], label='MAE Validación', linewidth=2)
ax2.set_xlabel('Época')
ax2.set_ylabel('MAE')
ax2.set_title('Error Absoluto Medio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Predicciones vs Valores Reales (Validación)
ax3 = axes[1, 0]
x_val_range = range(len(y_val_original))
ax3.plot(x_val_range, y_val_original, label='Valor Real', marker='o', alpha=0.7)
ax3.plot(x_val_range, y_pred_val_original, label='Predicción', marker='s', alpha=0.7)
ax3.set_xlabel('Índice')
ax3.set_ylabel('Valor')
ax3.set_title('Predicciones vs Valores Reales (Validación)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Métricas finales
ax4 = axes[1, 1]
metricas_nombres = ['Exactitud\nEntr.', 'Exactitud\nVal.', 'RMSE\nEntr.', 'RMSE\nVal.']
metricas_valores = [train_accuracy, val_accuracy, rmse_train * 10, rmse_val * 10]
colores = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
ax4.bar(metricas_nombres, metricas_valores, color=colores)
ax4.set_ylabel('Valor')
ax4.set_title('Resumen de Métricas')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("grafico_entrenamiento.png", dpi=300, bbox_inches='tight')
print("✓ Gráficos guardados en: grafico_entrenamiento.png")
plt.show()

print("\n✅ Modelo entrenado y evaluado correctamente.")
