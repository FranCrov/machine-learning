import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def ejecutar():
    print("\n--- K Vecinos Más Cercanos (KNN) ---")
    print("Dataset: Wine (scikit-learn)")
    print("Variables predictoras: características químicas del vino")
    print("Variable a predecir: clase de vino\n")

    datos = load_wine()
    X = datos.data
    y = datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    print(f"Precisión (Accuracy): {precision:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        display_labels=datos.target_names, 
        cmap="Blues", 
        ax=ax
    )
    plt.title("Matriz de Confusión - KNN (Wine)")
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/knn.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/knn.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("=== Algoritmo Aplicado: K Vecinos Más Cercanos (KNN) ===\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("• Dataset elegido: Wine (scikit-learn)\n")
        f.write("• Resultados obtenidos:\n")
        f.write(f"  - Precisión (Accuracy): {precision:.4f}\n\n")
        f.write("• Comparación con lo esperado:\n")
        f.write("  Al tratar con un problema de clasificación, se espera que el algoritmo KNN agrupe los datos según las similitudes en las características químicas para predecir la categoría del vino. Una alta precisión en la diagonal de la matriz de confusión indica que las clases forman grupos bien definidos.\n\n")
        f.write("• Captura de resultados:\n")
        f.write(f"  [PEGAR AQUÍ LA CAPTURA DEL GRÁFICO: {ruta_grafico}]\n")
    
    print(f"Informe estructurado guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")