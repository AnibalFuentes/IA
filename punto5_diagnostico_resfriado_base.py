# Importamos las librerías necesarias
from sklearn.svm import SVC  # Importamos el clasificador SVC con RBF kernel
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

# Creamos el clasificador SVC con kernel RBF
modelo_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)

# Entrenamos el modelo con los datos de entrada X y las salidas esperadas y
modelo_rbf.fit(X, y)

# Mostramos los parámetros y el soporte vectorial aprendido
print("Soportes vectores:", modelo_rbf.support_)

# Predicciones con el modelo entrenado
predicciones_rbf = modelo_rbf.predict(X)

# Mostramos los resultados de las predicciones comparados con las salidas esperadas
print("\nResultados con RBF:")
for entrada, real, pred in zip(X, y, predicciones_rbf):
    print(f"Entrada: {entrada}, Esperada: {real}, Predicha: {pred}")
