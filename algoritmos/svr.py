import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def ejecutar():
    print("\n--- SVR (Máquinas de Vectores de Soporte para Regresión) ---")
    print("Dataset: Energy Efficiency (OpenML)")
    print("Variables predictoras: características del edificio (X1 a X8)")
    print("Variable a predecir: carga de calefacción (Heating Load)\n")

    # 1. Cargar el dataset
    datos = fetch_openml(data_id=1472, as_frame=True)
    df = datos.frame

    # El dataset trae dos posibles objetivos: Y1 (calefacción) e Y2 (refrigeración).
    # Usamos Y1 y descartamos Y2 para quedarnos con una sola variable a predecir.
    columnas_objetivo = [c for c in df.columns if c.upper() in ("Y1", "Y2")]
    objetivo = "Y1" if "Y1" in df.columns else columnas_objetivo[0]

    X = df.drop(columns=columnas_objetivo)
    y = df[objetivo].astype(float)

    # 2. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Escalar (fundamental para SVR, que es sensible a la escala de los datos)
    escalador_X = StandardScaler()
    X_train_esc = escalador_X.fit_transform(X_train)
    X_test_esc = escalador_X.transform(X_test)

    # 4. Entrenar el modelo
    modelo = SVR(kernel="rbf", C=100, gamma="scale")
    modelo.fit(X_train_esc, y_train)

    # 5. Predecir y calcular métricas
    y_pred = modelo.predict(X_test_esc)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    # 6. Graficar: valores reales vs. predichos
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="seagreen", alpha=0.7, label="Predicciones")
    minimo = min(y_test.min(), y_pred.min())
    maximo = max(y_test.max(), y_pred.max())
    plt.plot([minimo, maximo], [minimo, maximo], color="firebrick", linewidth=2, label="Predicción perfecta")
    plt.xlabel("Valor real (Heating Load)")
    plt.ylabel("Valor predicho")
    plt.title("SVR - Dataset Energy Efficiency")
    plt.legend()
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/svr.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/svr.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - SVR (Máquinas de Vectores de Soporte para Regresión)\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Energy Efficiency (OpenML)\n")
        f.write(f"Variable objetivo: {objetivo} (carga de calefacción)\n\n")
        f.write("Kernel: rbf, C=100, gamma=scale\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
    print(f"Informe guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")