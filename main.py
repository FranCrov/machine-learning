"""
Trabajo Práctico - Machine Learning
Menú principal de la aplicación.
"""

from algoritmos import logistica
from algoritmos import svr
from algoritmos import naive_bayes
from algoritmos import lineal_simple
from algoritmos import  lineal_multiple
from algoritmos import polinomial
from algoritmos import bosques_aleatorios_reg
from algoritmos import svm_clasificacion
from algoritmos import bosques_aleatorios_clas
from algoritmos import knn
from algoritmos import arbol_clasificacion
from algoritmos import arbol_regresion

def mostrar_menu():
    print("\n" + "=" * 50)
    print("   MENÚ - ALGORITMOS DE MACHINE LEARNING")
    print("=" * 50)
    print("REGRESIÓN")
    print(" 1. Regresión lineal simple")
    print(" 2. Regresión lineal múltiple")
    print(" 3. Regresión polinomial")
    print(" 4. SVR (vectores de soporte)")
    print(" 5. Árbol de decisión (regresión)")
    print(" 6. Bosque aleatorio (regresión)")
    print("CLASIFICACIÓN")
    print(" 7. Regresión logística")
    print(" 8. K vecinos más cercanos (KNN)")
    print(" 9. SVM (clasificación)")
    print("10. Naive Bayes")
    print("11. Árbol de decisión (clasificación)")
    print("12. Bosque aleatorio (clasificación)")
    print(" 0. Salir")
    print("=" * 50)

def main():
    var = True
    while var:
        mostrar_menu()
        opcion = input("Elegí una opción: ").strip()

        if opcion == "1":
            lineal_simple.ejecutar()
        elif opcion == "2":
            lineal_multiple.ejecutar()
        elif opcion == "3":
            polinomial.ejecutar()
        elif opcion == "4":
            svr.ejecutar()
        elif opcion == "5":
            arbol_regresion.ejecutar()
        elif opcion == "6":
            bosques_aleatorios_reg.ejecutar()
        elif opcion == "7":
            logistica.ejecutar()   
        elif opcion == "8":
            knn.ejecutar()
        elif opcion == "9":
            svm_clasificacion.ejecutar()
        elif opcion == "10":
            naive_bayes.ejecutar()
        elif opcion == "11":
            arbol_clasificacion.ejecutar()
        elif opcion == "12":
            bosques_aleatorios_clas.ejecutar()
        elif opcion == "0":
            print("¡Hasta luego!")
            var = False
        else:
            print("Opción inválida. Intentá de nuevo.")

if __name__ == "__main__":
    main()