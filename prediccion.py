import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

model = load_model("modelo_lstm.keras", compile=False)
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv("numeros.csv")
numeros = df["numero"].astype(str)

data = []
for n in numeros:
    data.extend([int(d) for d in n])

data = np.array(data).reshape(-1,1)
data = scaler.transform(data)

ultima = data[-5:].reshape(1,5,1)

predicciones = []
entrada_temporal = ultima.copy()

for i in range(4):
    pred = model.predict(entrada_temporal, verbose=0)
    digito = int(scaler.inverse_transform(pred)[0][0])
    predicciones.append(digito)
    
    # Actualizar la entrada para la siguiente predicción
    entrada_temporal = np.append(entrada_temporal[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

print("4 Dígitos probables:", ''.join(map(str, predicciones)))
