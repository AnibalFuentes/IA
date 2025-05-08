import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Datos de entrada: representaciones numéricas de las letras
letras = {
    'A': [2, 2, 4, 4, 10, 10, 4, 4, 2, 2],
    'B': [10, 10, 6, 6, 10, 10, 6, 6, 10, 10],
    'T': [2, 2, 2, 2, 10, 10, 2, 2, 2, 2]
}

# Preparar datos
X = np.array(list(letras.values()))
clases = list(letras.keys())

# Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(clases)
y_cat = to_categorical(y_encoded)

# Definición del modelo
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(clases), activation='softmax'))

# Compilación y entrenamiento
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=150, verbose=0)

# Predicción
nueva_letra = np.array([[2, 2, 2, 2, 10, 10, 2, 2, 2, 2]])
pred = model.predict(nueva_letra)

# Mostrar resultado
indice = np.argmax(pred)
letra_reconocida = encoder.inverse_transform([indice])[0]
print("Letra reconocida:", letra_reconocida)
