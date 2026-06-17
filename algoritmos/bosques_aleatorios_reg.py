import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def ejecutar():
    print("\n--- Bosques Aleatorios (Regresión) ---")
    print("Dataset: California Housing (scikit-learn)")
    print("Variables predictoras: MedInc, HouseAge, AveRooms (Características de la zona)")
    print("Variable a predecir: MedHouseValue (Precio medio de la vivienda)\n")

    # 1. Cargar el dataset de viviendas de California
    california = datasets.fetch_california_housing(as_frame=True)
    X = california.data[['MedInc', 'HouseAge', 'AveRooms']].values  # Usamos 3 columnas clave
    y = california.target.values  # Precio medio

    # 2. Separar en entrenamiento y prueba (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Inicializar y entrenar el Bosque Aleatorio
    # Usamos 100 árboles de decisión en paralelo (n_estimators=100)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    modelo.fit(X_train, y_train)

    # 4. Predecir y calcular métricas
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Métricas obtenidas:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f} (Explicación del {r2*100:.1f}% de la varianza)")

    # 5. Graficar: Valores Reales vs. Predicciones
    # En regresión múltiple no se puede hacer una línea simple, se usa un gráfico de dispersión de acierto
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="forestgreen", label="Predicciones")
    
    # Línea de referencia perfecta (Donde y_test == y_pred)
    limites = [y_test.min(), y_test.max()]
    plt.plot(limites, limites, color="red", linestyle="--", linewidth=2, label="Predicción Perfecta")
    
    plt.xlabel("Valores Reales (Precio)")
    plt.ylabel("Valores Predichos por el Bosque")
    plt.title("Random Forest Regressor - California Housing")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 6. Guardar gráfico
    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/bosques_aleatorios_reg.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    # 7. Guardar informe en TXT
    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/bosques_aleatorios_reg.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Bosques Aleatorios (Regresión)\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: California Housing (scikit-learn)\n")
        f.write("Configuración: n_estimators=100, max_depth=10\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
    print(f"Informe escrito en: {ruta_informe}")