import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def ejecutar():
    print("\n--- Regresión Polinomial ---")
    print("Dataset: Sintético No Lineal (Generado con NumPy)")
    print("Variable predictora: X (Característica continua)")
    print("Variable a predecir: y (Resultado con ruido cuadrático)\n")

    # 1. Generar un dataset sintético curvo (No lineal)
    np.random.seed(42)
    X = 2 - 3 * np.random.normal(0, 1, 100)
    # Creamos una relación cuadrática (con X al cuadrado) y le sumamos ruido
    y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)
    
    # Reestructuramos X para que scikit-learn lo acepte como matriz (100 filas, 1 columna)
    X = X[:, np.newaxis]

    # 2. Separar en entrenamiento y prueba (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Transformar los datos para incluir características polinomiales (Grado 2)
    # Esto transforma nuestro [X] en [1, X, X^2] para que la regresión lineal pueda curvarse
    grado = 2
    pf = PolynomialFeatures(degree=grado)
    X_train_poli = pf.fit_transform(X_train)
    X_test_poli = pf.transform(X_test)

    # 4. Entrenar el modelo de Regresión Lineal sobre los datos transformados
    modelo = LinearRegression()
    modelo.fit(X_train_poli, y_train)

    # 5. Predecir y calcular métricas
    y_pred = modelo.predict(X_test_poli)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Métricas obtenidas:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    # 6. Graficar la curva de predicción de forma suave
    plt.figure(figsize=(8, 6))
    # Dibujamos los puntos reales de prueba
    plt.scatter(X_test, y_test, color="steelblue", label="Datos reales (Test)")
    
    # Para que la línea de la predicción quede una curva perfecta y no un zig-zag, 
    # ordenamos los puntos de X de menor a mayor antes de dibujar la línea
    X_ordenado = np.sort(X_test, axis=0)
    X_ordenado_poli = pf.transform(X_ordenado)
    y_ordenado_pred = modelo.predict(X_ordenado_poli)
    
    plt.plot(X_ordenado, y_ordenado_pred, color="firebrick", linewidth=2, label=f"Curva Polinomial (Grado {grado})")
    
    plt.xlabel("Variable Predictora (X)")
    plt.ylabel("Variable Objetivo (y)")
    plt.title("Regresión Polinomial - Dataset Sintético")
    plt.legend()
    plt.tight_layout()

    # 7. Guardar gráfico
    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/polinomial.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    # 8. Guardar informe en TXT
    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/polinomial.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Regresión Polinomial\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Sintético No Lineal (NumPy)\n")
        f.write(f"Grado del polinomio: {grado}\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
    print(f"Informe escrito en: {ruta_informe}")