import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def ejecutar():
    print("\n--- Regresión Lineal Simple ---")
    print("Dataset: Diabetes (scikit-learn)")
    print("Variable predictora: índice de masa corporal (bmi)")
    print("Variable a predecir: progresión de la enfermedad\n")

    # 1. Cargar el dataset
    datos = load_diabetes()
    X = datos.data[:, np.newaxis, 2]  # columna 'bmi' (índice 2)
    y = datos.target

    # 2. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 4. Predecir y calcular métricas
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Coeficiente (pendiente): {modelo.coef_[0]:.2f}")
    print(f"Intercepto: {modelo.intercept_:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    # 5. Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color="steelblue", label="Datos reales")
    plt.plot(X_test, y_pred, color="firebrick", linewidth=2, label="Predicción")
    plt.xlabel("Índice de masa corporal (normalizado)")
    plt.ylabel("Progresión de la enfermedad")
    plt.title("Regresión Lineal Simple - Dataset Diabetes")
    plt.legend()
    plt.tight_layout()

    # 6. Guardar gráfico e informe
    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/lineal_simple.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/lineal_simple.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Regresión Lineal Simple\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Diabetes (scikit-learn)\n")
        f.write("Variable predictora: bmi (índice de masa corporal)\n")
        f.write("Variable objetivo: progresión de la enfermedad\n\n")
        f.write(f"Coeficiente (pendiente): {modelo.coef_[0]:.4f}\n")
        f.write(f"Intercepto: {modelo.intercept_:.4f}\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
    print(f"Informe guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")