import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def ejecutar():
    print("\n--- Bosques Aleatorios (Clasificación) ---")
    print("Dataset: Wine (scikit-learn)")
    print("Variables predictoras: 13 análisis químicos (Alcohol, Ácido málico, Ceniza, etc.)")
    print("Variable a predecir: Tipo de cultivar de vino (Clase 0, Clase 1, Clase 2)\n")

    # 1. Cargar el dataset de Vinos
    vinos = datasets.load_wine()
    X = vinos.data
    y = vinos.target

    # 2. Separar en entrenamiento y prueba (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Inicializar y entrenar el Bosque Aleatorio para Clasificación
    # Usamos 100 árboles de decisión para clasificar las categorías
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Predecir y calcular métricas
    y_pred = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, y_pred)

    print(f"Métricas obtenidas:")
    print(f"  Exactitud Global (Accuracy): {exactitud * 100:.2f}%\n")
    print("  Reporte de Clasificación detallado:")
    print(classification_report(y_test, y_pred, target_names=vinos.target_names))

    # 5. Gráfico: Matriz de Confusión (En español las etiquetas fijas del gráfico)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=vinos.target_names)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Reds, ax=ax)  # Usamos Reds (tonos rojos/vino) para que quede temático
    plt.title("Matriz de Confusión - Random Forest Clasificación")
    plt.tight_layout()

    # 6. Guardar gráfico
    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/bosques_aleatorios_clas.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    # 7. Guardar informe en TXT
    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/bosques_aleatorios_clas.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Bosques Aleatorios (Clasificación)\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Wine (scikit-learn)\n")
        f.write(f"Exactitud (Accuracy): {exactitud * 100:.2f}%\n")
        f.write("\nReporte Detallado:\n")
        f.write(classification_report(y_test, y_pred, target_names=vinos.target_names))
    print(f"Informe escrito en: {ruta_informe}")