Sistema de Recomendación MovieLens:

Este proyecto implementa un sistema de recomendación basado en filtrado colaborativo utilizando el dataset ml-latest-small. El objetivo es procesar 100,836 calificaciones de 610 usuarios sobre 9,742 películas sin el uso de librerías de alto nivel para el cálculo de distancias.

1. Análisis de los Datos

El dataset se compone de cuatro archivos principales, pero para el sistema de recomendación y cálculo de distancias, el núcleo es ratings.csv.

Estructura de Calificaciones (ratings.csv)

userId: Identificador único (1 a 610).

movieId: Identificador de la película.

rating: Valor numérico (0.5 a 5.0).

timestamp: Tiempo de la calificación (no relevante para distancias básicas).

El Problema de la Dispersión (Sparsity)

Contamos con:

Total de combinaciones posibles: $610 \text{ usuarios} \times 9,742 \text{ películas} = 5,942,620$ celdas.

Calificaciones reales: $100,836$.

Densidad: $\approx 1.7\%$.

Esto significa que el 98.3% de la matriz de usuarios-películas está vacía. Intentar cargar una matriz completa de $610 \times 9742$ sería ineficiente en memoria y computacionalmente costoso.

2. Estrategias de Carga de Datos

Para optimizar el rendimiento sin usar librerías como Pandas, se proponen las siguientes estrategias:

A. Estructura de Diccionario Anidado (Hash Maps)

En lugar de una matriz densa, cargaremos los datos en un diccionario de diccionarios. Esta es la forma más eficiente de realizar búsquedas en Python.

# Estructura recomendada

prefs = {
'userId_1': {
'movieId_10': 4.0,
'movieId_15': 3.0
},
'userId_2': {
'movieId_10': 5.0,
'movieId_22': 2.5
}
}

Ventaja: El cálculo de la distancia entre el usuario $U$ y $V$ solo requiere iterar sobre las llaves del usuario con menos calificaciones, reduciendo la complejidad de $O(I_{total})$ a $O(I_{comun})$.

B. Indexación Inversa

Para la tarea de Recomendación, es útil tener un índice de películas para saber qué usuarios vieron qué película rápidamente.
pelicula_vistas_por = { 'movieId_10': [user1, user2, ...], ... }

C. Generador de Carga (Memory-Safe)

Si el dataset fuera más grande (ej. 25 millones de registros), usaríamos generadores para leer el CSV línea por línea sin cargar todo en RAM:

def load*ratings(path):
with open(path, 'r') as f:
next(f) # Saltar header
for line in f:
uid, mid, rating, * = line.split(',')
yield uid, mid, float(rating)

3. Planificación de Tareas Técnicas

Tarea 1: Algoritmo k-NN

Métricas a implementar: Manhattan y Euclidiana.

Optimización: Antes de calcular la distancia, usaremos set.intersection() sobre las llaves de los diccionarios de cada usuario para encontrar las películas comunes en tiempo $O(min(len(U), len(V)))$.

Tarea 2: Recomendación con Umbral

Lógica: Identificar ítems que los vecinos cercanos calificaron con $>3$ y que el usuario objetivo no ha consumido.

Puntuación: Aplicar promedio ponderado: $Pred(u, i) = \frac{\sum (Sim(u,v) \times R_{v,i})}{\sum |Sim(u,v)|}$.

Tarea 3: Influencers

Definición: Un "Influencer" será un perfil sintético creado a partir del promedio de calificaciones de los $k$ usuarios más representativos de un nicho.

Análisis: Se medirá cómo la inserción de este usuario afecta las recomendaciones de los demás.

Tarea 4: Complejidad Computacional

Se realizará un benchmarking midiendo el tiempo de ejecución ($T$) en función de:

Cantidad de registros: $[10k, 50k, 100k]$.

Algoritmo: Manhattan vs Euclidiana.

Estructura: Lista de listas vs Diccionarios.

4. Consideraciones de Paralelismo

Dado que el cálculo de distancia de un usuario contra todos los demás es independiente para cada par, utilizaremos multiprocessing.Pool para distribuir los 610 usuarios entre los núcleos disponibles, lo que debería resultar en una mejora de rendimiento de casi el $N \times \text{núcleos}$.
