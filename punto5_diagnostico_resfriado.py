# Importamos las librerías necesarias
from sklearn.linear_model import Perceptron  # Importamos el modelo Perceptrón de scikit-learn
import numpy as np  # Para manejar matrices y arrays de manera eficiente

# Datos de entrada (X1, X2, X3, X4: síntomas)
X = np.array([
    [0, 0, 0, 0],  # Dolor de cabeza: No, Fiebre: No, Tos: No, Dolor de rodilla: No
    [1, 1, 1, 1],  # Dolor de cabeza: Sí, Fiebre: Sí, Tos: Sí, Dolor de rodilla: Sí
    [1, 1, 1, 0],  # Dolor de cabeza: Sí, Fiebre: Sí, Tos: Sí, Dolor de rodilla: No
    [0, 0, 0, 1],  # Dolor de cabeza: No, Fiebre: No, Tos: No, Dolor de rodilla: Sí
    [0, 1, 1, 0],  # Dolor de cabeza: No, Fiebre: Sí, Tos: Sí, Dolor de rodilla: No
    [0, 1, 1, 1],  # Dolor de cabeza: No, Fiebre: Sí, Tos: Sí, Dolor de rodilla: Sí
    [0, 0, 1, 0],  # Dolor de cabeza: No, Fiebre: No, Tos: Sí, Dolor de rodilla: No
    [0, 0, 1, 1],  # Dolor de cabeza: No, Fiebre: No, Tos: Sí, Dolor de rodilla: Sí
    [1, 0, 1, 0],  # Dolor de cabeza: Sí, Fiebre: No, Tos: Sí, Dolor de rodilla: No
    [1, 0, 1, 1],  # Dolor de cabeza: Sí, Fiebre: No, Tos: Sí, Dolor de rodilla: Sí
])

# Salidas esperadas (y): 0 significa no tener resfriado, 1 significa tener resfriado
y = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])

# Creamos el modelo de Perceptrón
modelo = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# Entrenamos el modelo con los datos de entrada X y las salidas esperadas y
modelo.fit(X, y)

# Mostramos los pesos aprendidos por el modelo después del entrenamiento
# 'coef_' nos da los pesos aprendidos para cada entrada (síntoma)
print("Pesos aprendidos:", modelo.coef_)
# 'intercept_' nos da el valor del bias (desviación) aprendido
print("Bias aprendido:", modelo.intercept_)

# Ahora, utilizamos el modelo entrenado para hacer predicciones con los mismos datos de entrada
predicciones = modelo.predict(X)

# Mostramos los resultados de las predicciones comparados con las salidas esperadas
print("\nResultados:")
# Usamos zip() para recorrer simultáneamente X, y (salidas esperadas) y las predicciones obtenidas
for entrada, real, pred in zip(X, y, predicciones):
    print(f"Entrada: {entrada}, Esperada: {real}, Predicha: {pred}")
