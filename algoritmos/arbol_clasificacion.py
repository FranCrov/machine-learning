import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def ejecutar():
    print("\n--- Árbol de Decisión (Clasificación) ---")
    print("Dataset: Iris (scikit-learn)")
    print("Variables predictoras: medidas de pétalos y sépalos")
    print("Variable a predecir: especie de la flor\n")

    datos = load_iris()
    X = datos.data
    y = datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    print(f"Precisión (Accuracy): {precision:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        display_labels=datos.target_names, 
        cmap="Greens", 
        ax=ax
    )
    plt.title("Matriz de Confusión - Árbol de Decisión (Iris)")
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/arbol_clasificacion.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/arbol_clasificacion.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("=== Algoritmo Aplicado: Árbol de Decisión (Clasificación) ===\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("• Dataset elegido: Iris (scikit-learn)\n")
        f.write("• Resultados obtenidos:\n")
        f.write(f"  - Precisión (Accuracy): {precision:.4f}\n\n")
        f.write("• Comparación con lo esperado:\n")
        f.write("  Se esperaba que el modelo generara divisiones lógicas basadas en las dimensiones de las flores para clasificarlas en su especie correcta. Una alta precisión y una diagonal fuerte en la matriz de confusión indican que el árbol logró encontrar reglas efectivas sin sobreajustarse a los datos de entrenamiento.\n\n")
        f.write("• Captura de resultados:\n")
        f.write(f"  [PEGAR AQUÍ LA CAPTURA DEL GRÁFICO: {ruta_grafico}]\n")
    
    print(f"Informe estructurado guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")