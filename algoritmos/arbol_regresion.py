import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def ejecutar():
    print("\n--- Árbol de Decisión (Regresión) ---")
    print("Dataset: Diabetes (scikit-learn)")
    print("Variables predictoras: todas las características del paciente")
    print("Variable a predecir: progresión de la enfermedad\n")

    datos = load_diabetes()
    X = datos.data
    y = datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = DecisionTreeRegressor(max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="purple", alpha=0.6, label="Predicciones del Árbol")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="orange", linewidth=2, label="Predicción Perfecta")
    plt.xlabel("Progresión Real")
    plt.ylabel("Progresión Predicha")
    plt.title("Árbol de Decisión (Regresión) - Dataset Diabetes")
    plt.legend()
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/arbol_regresion.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/arbol_regresion.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("=== Algoritmo Aplicado: Árbol de Decisión (Regresión) ===\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("• Dataset elegido: Diabetes (scikit-learn)\n")
        f.write("• Resultados obtenidos:\n")
        f.write(f"  - MSE:  {mse:.4f}\n")
        f.write(f"  - RMSE: {rmse:.4f}\n")
        f.write(f"  - R2:   {r2:.4f}\n\n")
        f.write("• Comparación con lo esperado:\n")
        f.write("  Se esperaba que el Árbol de Decisión pudiera capturar relaciones no lineales entre las variables de los pacientes y la progresión de la diabetes mediante subdivisiones de los datos. Al comparar este R2 con el de una regresión lineal simple, podemos evaluar si el enfoque basado en reglas (árboles) funciona mejor para este conjunto de datos que el enfoque basado en una línea recta matemática.\n\n")
        f.write("• Captura de resultados:\n")
        f.write(f"  [PEGAR AQUÍ LA CAPTURA DEL GRÁFICO: {ruta_grafico}]\n")
    
    print(f"Informe estructurado guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")