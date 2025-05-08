import numpy as np

# Datos de entrada
X = np.array([
    [0.1, 0.9, 0.1],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.1],
    [0.9, 0.9, 0.9]
])

# Salidas deseadas
D = np.array([
    [0.1, 0.9],
    [0.1, 0.9],
    [0.1, 0.1],
    [0.9, 0.9]
])

# Parámetros
lr = 0.1
epochs = 100
threshold = 0.5
np.random.seed(1)
weights = np.random.uniform(-0.1, 0.1, (2, 3))  # 2 neuronas x 3 entradas

def activation(net):
    return (net >= threshold).astype(float)

for epoch in range(epochs):
    for i in range(len(X)):
        x = X[i]
        d = D[i]
        net = weights @ x
        y = activation(net)
        error = d - y
        weights += lr * np.outer(error, x)

print("Pesos entrenados (Perceptrón):")
print(weights)
