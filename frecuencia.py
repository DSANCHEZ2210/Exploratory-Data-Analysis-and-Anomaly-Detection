from collections import Counter
import re

# Ruta del archivo (cámbiala si es necesario)
archivo = 'anomalias.txt'

def leer_y_procesar_archivo(ruta_archivo):
    combinaciones = []

    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            # Elimina el emoji y la fecha/hora
            partes = linea.split('Anomalías en')
            if len(partes) < 2:
                continue  # Si no hay anomalías, sigue con la siguiente línea

            # Extrae la lista de variables y limpia espacios
            variables_str = partes[1].strip()

            # Convierte la lista en un conjunto ordenado para evitar duplicados con diferente orden
            variables_lista = [var.strip() for var in variables_str.split(',')]
            variables_ordenadas = tuple(sorted(variables_lista))

            combinaciones.append(variables_ordenadas)

    return combinaciones

def contar_combinaciones(combinaciones):
    contador = Counter(combinaciones)
    return contador

def mostrar_resultados(contador):
    print("\nCombinaciones de variables de anomalías más frecuentes:\n")
    for combinacion, frecuencia in contador.most_common():
        print(f"Ocurrió {frecuencia} veces:\n  {combinacion}\n")

# --------- PROGRAMA PRINCIPAL ---------
combinaciones = leer_y_procesar_archivo(archivo)
contador = contar_combinaciones(combinaciones)
mostrar_resultados(contador)
