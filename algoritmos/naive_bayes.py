import os
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def ejecutar():
    print("\n--- Naive Bayes ---")
    print("Dataset: Wine (scikit-learn)")
    print("Variables predictoras: 13 características químicas del vino")
    print("Variable a predecir: tipo de vino (3 clases)\n")

    # 1. Cargar el dataset
    datos = load_wine()
    X = datos.data
    y = datos.target

    # 2. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Entrenar el modelo (GaussianNB: asume que cada característica sigue
    #    una distribución normal dentro de cada clase; no requiere escalado)
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    # 4. Predecir y calcular métricas
    y_pred = modelo.predict(X_test)

    # average="macro" porque hay 3 clases, no 2 como en los algoritmos anteriores
    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    matriz = confusion_matrix(y_test, y_pred)

    print(f"Exactitud (accuracy): {exactitud:.4f}")
    print(f"Precisión (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print("\nMatriz de confusión:")
    print(matriz)

    # 5. Graficar la matriz de confusión (ahora es 3x3, una clase por tipo de vino)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=datos.target_names)
    disp.plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_title("Matriz de Confusión - Naive Bayes (Wine)")
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/naive_bayes.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/naive_bayes.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Naive Bayes\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Wine (scikit-learn)\n")
        f.write("Variable objetivo: tipo de vino (3 clases)\n\n")
        f.write(f"Exactitud:  {exactitud:.4f}\n")
        f.write(f"Precisión:  {precision:.4f}\n")
        f.write(f"Recall:     {recall:.4f}\n")
        f.write(f"F1-score:   {f1:.4f}\n")
        f.write(f"\nMatriz de confusión:\n{matriz}\n")
    print(f"Informe guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")