import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import matplotlib.pyplot as plt

# Datos binarios (síntomas) y etiquetas (resfriado o no)
X_train = np.array([
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1]
])
y_train = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])

# -------------------------------------
# Funciones RBF
# -------------------------------------
def rbf_features(X, centers, gamma):
    return rbf_kernel(X, centers, gamma=gamma)

def train_rbf(X, y, n_centers=4, gamma=1.0):
    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    Phi = rbf_features(X, centers, gamma)
    W = np.linalg.pinv(Phi).dot(y)
    return W, centers

def predict_rbf(X_new, W, centers, gamma):
    Phi = rbf_features(X_new, centers, gamma)
    return Phi.dot(W)

# -------------------------------------
# Diagnóstico para una entrada
# -------------------------------------
sintomas = ["Dolor de cabeza", "Fiebre", "Tos", "Dolor de rodilla"]
entrada_ejemplo = [1, 0, 0, 0]

def diagnosticar_resfriado_rbf(entrada_binaria, W, centers, gamma):
    entrada = np.array([entrada_binaria])
    pred = predict_rbf(entrada, W, centers, gamma)
    return 1 if pred[0] > 0.5 else 0, pred[0]

# -------------------------------------
# Comparar modelos con diferentes centros
# -------------------------------------
def evaluar_modelo(n_centros, gamma=1.0):
    W, centers = train_rbf(X_train, y_train, n_centros, gamma)
    valores_pred = predict_rbf(X_train, W, centers, gamma)
    pred_bin = (valores_pred > 0.5).astype(int)
    accuracy = np.mean(pred_bin == y_train)
    
    return {
        "centros": n_centros,
        "accuracy": accuracy,
        "real": y_train,
        "predicho": pred_bin,
        "valores_rbf": valores_pred
    }

resultados_4 = evaluar_modelo(4)
resultados_7 = evaluar_modelo(5)

# -------------------------------------
# Mostrar resultados en consola
# -------------------------------------
def mostrar_resultados(resultados):
    print(f"Modelo con {resultados['centros']} centros")
    print(f"Precisión: {resultados['accuracy']:.2f}")
    print("\nValores predichos vs reales:")
    for i, (real, pred, valor_rbf) in enumerate(zip(resultados["real"], resultados["predicho"], resultados["valores_rbf"])):
        print(f"Ejemplo {i+1}: Real = {real}, Predicho = {pred}, Valor RBF = {valor_rbf:.4f}")
    print("\n" + "-"*40)

mostrar_resultados(resultados_4)
mostrar_resultados(resultados_7)

# -------------------------------------
# Mostrar resultados y gráficas de dispersión
# -------------------------------------
def mostrar_grafica_dispersion(resultados, titulo):
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(resultados["real"])), resultados["real"], label="Real", color="blue", marker='o', s=100, alpha=0.7)
    plt.scatter(np.arange(len(resultados["predicho"])), resultados["predicho"], label="Predicho", color="red", marker='x', s=100, alpha=0.7)
    plt.title(f"{titulo} - Precisión: {resultados['accuracy']:.2f}")
    plt.xlabel("Ejemplo")
    plt.ylabel("Etiqueta")
    plt.legend()
    plt.grid(True)

# Mostrar las gráficas de dispersión
mostrar_grafica_dispersion(resultados_4, "Modelo RBF con 4 centros")
mostrar_grafica_dispersion(resultados_7, "Modelo RBF con 7 centros")

plt.show()


