import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============ 1. VALIDACIÓN DE DATOS ============
df = pd.read_csv("numeros.csv")

# Validar que la columna existe
if "numero" not in df.columns:
    print("Error: La columna 'numero' no existe en el CSV")
    exit()

# Validar que no hay valores nulos
if df["numero"].isnull().any():
    print("Advertencia: Se encontraron valores nulos, removiendo...")
    df = df.dropna()

todos = "".join(df["numero"].astype(str))

# Validar que solo contiene dígitos
if not todos.isdigit():
    print("Advertencia: Se encontraron caracteres no numéricos")
    todos = ''.join(c for c in todos if c.isdigit())

print(f"✓ Total de dígitos validados: {len(todos)}\n")

# ============ 2. FRECUENCIA CON PORCENTAJES ============
frecuencia = Counter(todos)
total_digitos = len(todos)

print("=" * 50)
print("FRECUENCIA DE DÍGITOS")
print("=" * 50)
print(f"{'Dígito':<8} {'Cantidad':<12} {'Porcentaje':<12}")
print("-" * 50)

frecuencia_datos = []
for d in sorted(frecuencia.keys()):
    f = frecuencia[d]
    porcentaje = (f / total_digitos) * 100
    print(f"{d:<8} {f:<12} {porcentaje:>6.2f}%")
    frecuencia_datos.append({"Dígito": d, "Cantidad": f, "Porcentaje": porcentaje})

print()

# ============ 3. ESTADÍSTICAS DETALLADAS ============
digitos_numeros = [int(d) for d in todos]

print("=" * 50)
print("ESTADÍSTICAS")
print("=" * 50)
print(f"Media: {np.mean(digitos_numeros):.2f}")
print(f"Mediana: {np.median(digitos_numeros):.2f}")
print(f"Desv. Estándar: {np.std(digitos_numeros):.2f}")
print(f"Mín: {np.min(digitos_numeros)}")
print(f"Máx: {np.max(digitos_numeros)}")
print(f"Rango: {np.max(digitos_numeros) - np.min(digitos_numeros)}")
print()

# ============ 4. ANÁLISIS DE TRANSICIONES (Secuencias) ============
print("=" * 50)
print("ANÁLISIS DE TRANSICIONES (Qué dígito sigue a cuál)")
print("=" * 50)

transiciones = {}
for i in range(len(todos) - 1):
    par = todos[i] + "->" + todos[i+1]
    transiciones[par] = transiciones.get(par, 0) + 1

transiciones_ordenadas = sorted(transiciones.items(), key=lambda x: x[1], reverse=True)
print(f"Top 15 transiciones más frecuentes:\n")
print(f"{'Transición':<12} {'Cantidad':<12} {'Porcentaje':<12}")
print("-" * 50)

for transicion, count in transiciones_ordenadas[:15]:
    porcentaje = (count / (total_digitos - 1)) * 100
    print(f"{transicion:<12} {count:<12} {porcentaje:>6.2f}%")

print()

# ============ 5. EXPORTAR RESULTADOS A CSV ============
df_frecuencia = pd.DataFrame(frecuencia_datos)
df_frecuencia.to_csv("analisis_frecuencia.csv", index=False)
print("✓ Frecuencias exportadas a: analisis_frecuencia.csv")

df_transiciones = pd.DataFrame([
    {"Transición": k, "Cantidad": v, "Porcentaje": (v/(total_digitos-1))*100} 
    for k, v in transiciones_ordenadas
])
df_transiciones.to_csv("analisis_transiciones.csv", index=False)
print("✓ Transiciones exportadas a: analisis_transiciones.csv\n")

# ============ 6. VISUALIZACIÓN CON GRÁFICOS ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis de Patrones de Números", fontsize=16, fontweight='bold')

# Gráfico 1: Frecuencia de dígitos
ax1 = axes[0, 0]
digitos = sorted(frecuencia.keys())
cantidades = [frecuencia[d] for d in digitos]
colors = plt.cm.viridis(np.linspace(0, 1, len(digitos)))
ax1.bar(digitos, cantidades, color=colors)
ax1.set_xlabel("Dígito")
ax1.set_ylabel("Frecuencia")
ax1.set_title("Frecuencia de Dígitos")
ax1.grid(axis='y', alpha=0.3)

# Gráfico 2: Porcentajes
ax2 = axes[0, 1]
porcentajes = [(frecuencia[d] / total_digitos) * 100 for d in digitos]
ax2.pie(cantidades, labels=digitos, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title("Distribución Porcentual")

# Gráfico 3: Top 10 transiciones
ax3 = axes[1, 0]
top_transiciones = transiciones_ordenadas[:10]
trans_labels = [t[0] for t in top_transiciones]
trans_counts = [t[1] for t in top_transiciones]
ax3.barh(range(len(trans_labels)), trans_counts, color=plt.cm.plasma(np.linspace(0, 1, len(trans_labels))))
ax3.set_yticks(range(len(trans_labels)))
ax3.set_yticklabels(trans_labels)
ax3.set_xlabel("Frecuencia")
ax3.set_title("Top 10 Transiciones Más Frecuentes")
ax3.grid(axis='x', alpha=0.3)

# Gráfico 4: Distribución acumulada
ax4 = axes[1, 1]
digitos_ord = sorted(digitos)
frecuencia_ord = [frecuencia[d] for d in digitos_ord]
acumulada = np.cumsum(frecuencia_ord)
ax4.plot(digitos_ord, acumulada, marker='o', linewidth=2, markersize=8)
ax4.fill_between(range(len(digitos_ord)), acumulada, alpha=0.3)
ax4.set_xlabel("Dígito")
ax4.set_ylabel("Frecuencia Acumulada")
ax4.set_title("Distribución Acumulada")
ax4.grid(True, alpha=0.3)
ax4.set_xticks(digitos_ord)

plt.tight_layout()
plt.savefig("analisis_patrones.png", dpi=300, bbox_inches='tight')
print("✓ Gráficos guardados en: analisis_patrones.png")
plt.show()

print("\n" + "=" * 50)
print("✓ Análisis completado exitosamente")
print("=" * 50)
