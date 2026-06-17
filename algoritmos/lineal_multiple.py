import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def ejecutar():
    print("\n--- Regresión Lineal Múltiple ---")
    print("Dataset: California Housing (scikit-learn)")
    print("Variables predictoras: múltiples (ingreso, antigüedad, habitaciones, etc.)")
    print("Variable a predecir: precio de la vivienda\n")

    datos = fetch_california_housing()
    X = datos.data
    y = datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="forestgreen", alpha=0.4, label="Predicciones")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2, label="Predicción Perfecta")
    plt.xlabel("Precio Real")
    plt.ylabel("Precio Predicho")
    plt.title("Regresión Lineal Múltiple - California Housing")
    plt.legend()
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/lineal_multiple.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/lineal_multiple.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("=== Algoritmo Aplicado: Regresión Lineal Múltiple ===\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("• Dataset elegido: California Housing (scikit-learn)\n")
        f.write("• Resultados obtenidos:\n")
        f.write(f"  - MSE:  {mse:.4f}\n")
        f.write(f"  - RMSE: {rmse:.4f}\n")
        f.write(f"  - R2:   {r2:.4f}\n\n")
        f.write("• Comparación con lo esperado:\n")
        f.write("  Al utilizar múltiples características (como ingreso medio y antigüedad) se esperaba obtener una predicción más ajustada a la realidad que con una sola variable. El R2 refleja qué porcentaje de la varianza en los precios logra explicar este modelo conjunto.\n\n")
        f.write("• Captura de resultados:\n")
        f.write(f"  [PEGAR AQUÍ LA CAPTURA DEL GRÁFICO: {ruta_grafico}]\n")
    
    print(f"Informe estructurado guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")