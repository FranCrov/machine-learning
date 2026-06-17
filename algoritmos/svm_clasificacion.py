import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def ejecutar():
    print("\n--- Máquinas de Vectores de Soporte (SVM - Clasificación) ---")
    print("Dataset: Breast Cancer Wisconsin (scikit-learn)")
    print("Variables predictoras: Características geométricas de las células (radio, textura, etc.)")
    print("Variable a predecir: Diagnóstico (0 = Maligno, 1 = Benigno)\n")

    # 1. Cargar el dataset de Cáncer de Mama
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # 2. Separar en entrenamiento y prueba (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Inicializar y entrenar el modelo SVM
    # Usamos un kernel lineal para encontrar el hiperplano óptimo de separación
    modelo = SVC(kernel='linear', random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Predecir y calcular métricas de clasificación
    y_pred = modelo.predict(X_test)
    
    exactitud = accuracy_score(y_test, y_pred)
    
    print(f"Métricas obtenidas:")
    print(f"  Exactitud Global (Accuracy): {exactitud * 100:.2f}%\n")
    print("  Reporte de Clasificación detallado:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))

    # 5. Gráfico: Matriz de Confusión
    # En clasificación, el mejor gráfico es la matriz de confusión para ver en qué le erró
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Matriz de Confusión - SVM Clasificación")
    plt.tight_layout()

    # 6. Guardar gráfico
    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/svm_clasificacion.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    # 7. Guardar informe en TXT
    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/svm_clasificacion.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Máquinas de Vectores de Soporte (SVM Clasificación)\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Breast Cancer Wisconsin (scikit-learn)\n")
        f.write(f"Exactitud (Accuracy): {exactitud * 100:.2f}%\n")
        f.write("\nReporte Detallado:\n")
        f.write(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print(f"Informe escrito en: {ruta_informe}")