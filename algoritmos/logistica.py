import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def ejecutar():
    print("\n--- Regresión Logística ---")
    print("Dataset: Titanic (OpenML)")
    print("Variables predictoras: clase, sexo, edad, tarifa, etc.")
    print("Variable a predecir: sobrevivió (sí / no)\n")

    # 1. Cargar el dataset
    datos = fetch_openml(data_id=40945, as_frame=True)
    df = datos.frame

    # 2. Limpieza: descartamos columnas que no sirven para predecir o que
    #    filtran directamente la respuesta (boat/body indican casi con
    #    certeza si la persona sobrevivió, sería "hacer trampa").
    columnas_a_descartar = ["name", "ticket", "cabin", "room", "boat", "body", "home.dest"]
    df = df.drop(columns=[c for c in columnas_a_descartar if c in df.columns])

    # 3. Separar variable objetivo
    objetivo = "survived"
    y = pd.to_numeric(df[objetivo].astype(str), errors="coerce").astype(int)
    X = df.drop(columns=[objetivo])

    # 4. Codificar variables categóricas (sex, embarked, etc.) como dummies
    X = pd.get_dummies(X, drop_first=True)

    # 5. Imputar valores faltantes (age y fare suelen tener NaN)
    imputador = SimpleImputer(strategy="median")
    X_imputado = pd.DataFrame(imputador.fit_transform(X), columns=X.columns)

    # 6. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputado, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Escalar
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    # 8. Entrenar el modelo
    modelo = LogisticRegression(max_iter=10000)
    modelo.fit(X_train_esc, y_train)

    # 9. Predecir y calcular métricas
    y_pred = modelo.predict(X_test_esc)

    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matriz = confusion_matrix(y_test, y_pred)

    print(f"Exactitud (accuracy): {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad (recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nMatriz de confusión:")
    print(matriz)

    # 10. Graficar la matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=["No sobrevivió", "Sobrevivió"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Matriz de Confusión - Regresión Logística (Titanic)")
    plt.tight_layout()

    os.makedirs("graficos", exist_ok=True)
    ruta_grafico = "graficos/logistica.png"
    plt.savefig(ruta_grafico)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    plt.show()
    plt.close()

    os.makedirs("informes", exist_ok=True)
    ruta_informe = "informes/logistica.txt"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("INFORME - Regresión Logística\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Dataset: Titanic (OpenML)\n")
        f.write("Variable objetivo: sobrevivió (sí / no)\n\n")
        f.write(f"Exactitud:  {exactitud:.4f}\n")
        f.write(f"Precisión:  {precision:.4f}\n")
        f.write(f"Recall:     {recall:.4f}\n")
        f.write(f"F1-score:   {f1:.4f}\n")
        f.write(f"\nMatriz de confusión:\n{matriz}\n")
    print(f"Informe guardado en: {ruta_informe}")

    input("\nPresioná Enter para volver al menú...")