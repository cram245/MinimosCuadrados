# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# --- Datos ---
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
y = np.array([0.7, 0.45, 0.026, -0.22, -0.29, -0.3])

# --- Modelo y funciones auxiliares ---
def model(x, a, b, c):
    """p(x) = a + b * exp(-c * x^2)"""
    return a + b * np.exp(-c * x**2)

def loss_and_grads(x, y, a, b, c):
    """
    Calcula la pérdida (suma de cuadrados) y sus gradientes parciales.
    r_k = p(x_k) - y_k
    L = sum(r_k^2)
    """
    # Predicción y residuo
    E = np.exp(-c * x**2)
    p = a + b * E
    r = p - y
    
    # Pérdida
    L = np.sum(r**2)
    
    # Gradientes
    dL_da = 2 * np.sum(r * 1)
    dL_db = 2 * np.sum(r * E)
    dL_dc = 2 * np.sum(r * ( -b * x**2 * E ))
    
    return L, dL_da, dL_db, dL_dc

# --- Inicialización ---
a, b, c = y[-1], y[0] - y[-1], 1.0

# Hiperparámetros de descenso por gradiente
alpha = 1e-2     # tasa de aprendizaje
max_iter = int(1e6)  # número de iteraciones
tol = 1e-8       # tolerancia para parada temprana

# Historial para diagnóstico
history = {"loss": [], "a": [], "b": [], "c": []}

# --- Bucle de optimización ---
for i in range(int(max_iter)):
    L, g_a, g_b, g_c = loss_and_grads(x, y, a, b, c)
    
    # Guarda historial
    history["loss"].append(L)
    history["a"].append(a)
    history["b"].append(b)
    history["c"].append(c)
    
    # Actualiza parámetros
    a_new = a - alpha * g_a
    b_new = b - alpha * g_b
    c_new = c - alpha * g_c
    
    # Comprueba convergencia
    if abs(a_new - a) < tol and abs(b_new - b) < tol and abs(c_new - c) < tol:
        a, b, c = a_new, b_new, c_new
        print(f"Convergencia alcanzada en {i} iteraciones.")
        break
    
    a, b, c = a_new, b_new, c_new

else:
    print("Se alcanzó el máximo de iteraciones sin plena convergencia.")

# Resultado final
print(f"Parámetros ajustados:")
print(f"  a = {a:.6f}")
print(f"  b = {b:.6f}")
print(f"  c = {c:.6f}")

# --- Visualización del ajuste ---
xfine = np.linspace(0, 1, 200)
yfit = model(xfine, a, b, c)

plt.figure(figsize=(6,4))
plt.scatter(x, y, label="Datos", color="black")
plt.plot(xfine, yfit, label="Ajuste GD: $a + b e^{-c x^2}$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Ajuste manual por mínimos cuadrados (descenso por gradiente)")
plt.tight_layout()
plt.show()

# --- (Opcional) Evolución de la pérdida ---
plt.figure(figsize=(6,4))
plt.plot(history["loss"])
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Pérdida $L$")
plt.title("Convergencia del descenso por gradiente")
plt.tight_layout()
plt.show()
