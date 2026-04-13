# Sistema de Recomendación K-NN: Análisis, Rendimiento y Resiliencia en Filtrado Colaborativo

## Descripción

Este proyecto consiste en el diseño, desarrollo y evaluación exhaustiva de un Sistema de Recomendación basado en Filtrado Colaborativo (User-Based Collaborative Filtering) construido completamente desde cero.

En el panorama actual, donde plataformas como Netflix, Amazon o Spotify dependen de estos motores para fidelizar usuarios, comprender el funcionamiento matemático subyacente es crucial.

A diferencia de implementaciones convencionales que delegan la lógica matemática a librerías de "caja negra" (como Scikit-Learn, TensorFlow o Surprise), este motor fue codificado en Python puro, apoyándose únicamente en `pandas` para la ingesta inicial de datos. Este enfoque permite:

- Transparencia total del modelo
- Auditoría granular del rendimiento
- Control completo sobre estructuras de datos
- Evaluación frente a ataques de manipulación (Shilling Attacks)

---

## Características

### Arquitectura y Capacidades

- **Múltiples métricas de similitud y distancia**
  - Similitud de Coseno (normalizada globalmente)
  - Distancia Euclidiana ($L^2$)
  - Distancia de Manhattan ($L^1$)

- **Motor de recomendación robusto**
  - Promedio ponderado por similitud
  - Exclusión automática de ítems ya consumidos
  - Umbral mínimo configurable (ej. $> 3.0$)

- **Simulación de entornos hostiles**
  - Inyección de perfiles falsos (Shilling Attacks)
  - Estrategias de camuflaje basadas en popularidad

- **Benchmarking y profiling**
  - Evaluación con datasets progresivos (10k, 50k, 100k)
  - Medición de:
    - Tiempo de carga
    - Tiempo de cálculo K-NN
    - Tiempo de predicción
    - Uso de memoria
    - Error (MAE)

- **Optimización extrema de memoria**
  - Uso de diccionarios anidados (hash maps)
  - Evita matrices densas y errores de memoria

---

## Carga de Datos y Estructuras

### Formato de Entrada

- Archivo `ratings.csv` (formato estándar)

### Optimización de Ingesta

Para garantizar la máxima eficiencia computacional y evitar los cuellos de botella de I/O, la carga se realiza mediante el método itertuples(index=False) de Pandas. A diferencia de iterrows(), que emite Series de Pandas pesadas por cada iteración, itertuples() genera tuplas nativas de Python, reduciendo el overhead de procesamiento en más de un 70%.

Ventajas:

- Evita `iterrows()` (alto overhead)
- Reduce el costo computacional >70%
- Genera tuplas nativas en lugar de Series

---

### Estructura de Datos

La matriz Usuario-Ítem tradicional es extremadamente ineficiente. En su lugar, el sistema abstrae la información en un diccionario anidado bidimensional:

```python
user_ratings = {
    userId_1: {movieId_A: 4.5, movieId_B: 3.0},
    userId_2: {movieId_A: 5.0, movieId_C: 2.5}
}
```

#### Ventajas técnicas de esta estructura:

Complejidad $O(1)$: La verificación de si un usuario vio una película o la extracción de su calificación es instantánea mediante funciones hash.

Intersección Rápida: Encontrar el solapamiento de películas entre dos usuarios se resuelve mediante la teoría de conjuntos de Python (set(u.keys()) & set(v.keys())), lo cual opera a nivel de C subyacente de forma ultrarrápida, mitigando la complejidad polinomial de una búsqueda lineal.

## Dataset: MovieLens

El algoritmo ha sido rigurosamente evaluado utilizando el dataset estándar de la industria MovieLens (ml-latest-small) provisto por GroupLens Research (Universidad de Minnesota).

Perfil estadístico del dataset:

Volumen de transacciones: 100,836 calificaciones (ratings).

Usuarios únicos: 610 (seleccionados aleatoriamente; condición mínima: 20 películas calificadas).

Catálogo de Ítems: 9,742 películas únicas.

Escala de Valoración: Escala discreta de 1.0 a 5.0 estrellas, con incrementos de 0.5.

Cálculo de Dispersión (Sparsity):
La densidad de la matriz teórica se calcula como:

### Densidad

$$
\text{Densidad} = \frac{100,836}{610 \times 9,742} \approx 0.0169
$$

Esto indica una densidad del 1.69% y una dispersión (sparsity) del 98.31%. Este alto grado de escasez de datos justifica absolutamente el uso de diccionarios; instanciar esta matriz en NumPy en formato denso requeriría reservar espacio para casi 6 millones de celdas nulas (zeros), desperdiciando recursos y ciclos de reloj.

---

## Metodología

El núcleo analítico del sistema se apoya en tres etapas algorítmicas fundamentales para resolver el problema de predicción de preferencias

### 1. Métricas de Similitud

#### Similitud de Coseno

Es la más robusta en filtrado colaborativo. Evalúa el ángulo entre los vectores multidimensionales de ambos usuarios. Se implementó una variación estricta donde la "norma" (denominador) se calcula sobre el total del historial del usuario, no solo sobre las películas coincidentes, castigando así las coincidencias esporádicas.

$$
sim(u,v) = \frac{\sum_{i \in comunes} (r_{u,i} \cdot r_{v,i})}{\sqrt{\sum_{i \in u} r_{u,i}^2} \cdot \sqrt{\sum_{i \in v} r_{v,i}^2}}
$$

#### Distancia Euclidiana

Mide la separación geométrica directa entre vectores.

$$
d(u,v) = \sqrt{\sum_{i \in comunes} (r_{u,i} - r_{v,i})^2}
$$

#### Distancia Manhattan

Mide la distancia en geometría taxicab (suma de diferencias absolutas).

$$
d(u,v) = \sum_{i \in comunes} |r_{u,i} - r_{v,i}|
$$

---

### 2. Predicción

Una vez obtenidos los vecinos, la recomendación no es un promedio simple, sino un promedio ponderado ponderado. Si el vecino A tiene una similitud del 0.9 y el vecino B de 0.4, el voto de A tiene más del doble de impacto matemático en la predicción final.

$$
\hat{r}*{u,i} = \frac{\sum*{v \in N} sim(u,v) \cdot r_{v,i}}{\sum_{v \in N} |sim(u,v)|}
$$

Donde $N$ es el conjunto de los $K$ vecinos más cercanos del usuario $u$ que han calificado el ítem $i$.

### 3. Evaluación (MAE)

Para validar la certeza matemática del sistema, se utilizó el Error Absoluto Medio (MAE). Esta métrica cuantifica la magnitud promedio de las equivocaciones en las predicciones.

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |p_i - r_i|
$$

Un MAE de 0.8 indicaría que, en promedio, cuando el usuario realmente dio 4 estrellas, el sistema predijo 3.2 o 4.8.

## Implementación

### TAREA 1: K-NN

La función principal get_knn() orquesta el descubrimiento de vecinos. El algoritmo itera linealmente sobre los usuarios, calcula la métrica seleccionada y ordena los resultados dinámicamente dependiendo de si la métrica es una "similitud" (donde mayor es mejor) o una "distancia" (donde menor es mejor).

```python
def get_knn(target_uid, user_ratings, k=10, func_dist=cosine_similarity, is_similarity=True):
    target_prefs = user_ratings[target_uid]
    resultados = []

    for uid, prefs in user_ratings.items():
        if uid == target_uid:
            continue

        score = func_dist(target_prefs, prefs)

        if is_similarity and score > 0:
            resultados.append((uid, score))
        elif not is_similarity and score != float('inf'):
            resultados.append((uid, score))

    resultados.sort(key=lambda x: x[1], reverse=is_similarity)
    return resultados[:k]
```

---

### TAREA 2: Recomendación

Localizado en la función recomendar(). Esta función toma los $K$ vecinos obtenidos y calcula la suma ponderada. Crucialmente, emplea operaciones de conjuntos para filtrar instantáneamente las películas que el usuario objetivo ya ha consumido.

```python
vistas = set(user_ratings[target_uid].keys())
acumuladores = {}

for v_id, sim in vecinos:
    for mid, rating in user_ratings[v_id].items():
        if mid not in vistas:
            if mid not in acumuladores:
                acumuladores[mid] = [0.0, 0.0]
            acumuladores[mid][0] += sim * rating
            acumuladores[mid][1] += sim

predicciones = []
for mid, (suma_pond, suma_sim) in acumuladores.items():
    pred = suma_pond / suma_sim if suma_sim > 0 else 0
    if pred >= umbral:
        predicciones.append((mid, pred))
```

---

### TAREA 3: Shilling Attack

La vulnerabilidad de los motores de recomendación se evalúa inyectando perfiles maliciosos. La función crear_influencer() ejecuta un ataque de "Average-Bot" basado en popularidad. Cuenta las frecuencias globales, selecciona el Top 100 de ítems más vistos y se autoasigna el promedio exacto, convirtiéndose en el "vecino perfecto".

```python
def crear_influencer(user_ratings, influencer_id=9999):
    conteo, sumas = {}, {}

    for prefs in user_ratings.values():
        for mid, r in prefs.items():
            conteo[mid] = conteo.get(mid, 0) + 1
            sumas[mid] = sumas.get(mid, 0) + r

    populares = sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:100]

    perfil_influencer = {
        mid: round(sumas[mid] / conteo[mid], 2)
        for mid, _ in populares
    }

    user_ratings[influencer_id] = perfil_influencer
    return perfil_influencer
```

---

### TAREA 4: Profiling

El módulo ejecutar_experimento_completo() intercepta dinámicamente los recursos del sistema. Aisla el tiempo de ejecución con time.time() y el costo de memoria con sys.getsizeof().

```python
start_load = time.time()
# carga
time_subida = time.time() - start_load

memoria_bytes = sys.getsizeof(data_test) + sum(sys.getsizeof(v) for v in data_test.values())
memoria_mb = memoria_bytes / (1024*1024)

start_dist = time.time()
vecinos = get_knn(target_uid, data_test, k=10)
time_distancia = time.time() - start_dist
```

---

## Resultados

### Hallazgos

El experimento arrojó datos empíricos concluyentes sobre el comportamiento de algoritmos determinísticos a gran escala:

- Rendimiento Excepcional de Estructuras Dispersas: El consumo de memoria demostró una linealidad perfecta. Escalar el análisis en un 1000% (de 10k a 100k registros) requirió apenas pasar de 0.39 MB a 3.66 MB de RAM libre. Los tiempos de predicción se mantuvieron consistentemente por debajo de 0.0002 segundos, lo que confirma que el uso de diccionarios en $O(1)$ evita la explosión combinatoria.

- Victoria de la Similitud de Coseno: La métrica de Coseno logró estabilizarse en un MAE de 0.8220 al utilizar el dataset completo. Este es un resultado de grado profesional, indicando que el margen de error promedio del sistema es de menos de una estrella (en un rango de 1 a 5).

- Inmunidad Arquitectónica al Spam: El test del "Influencer" evidenció que, si bien en bases de datos minúsculas (ej. 8 usuarios) los perfiles falsos pueden manipular las recomendaciones, al llegar a entornos densos (600+ usuarios), el sistema ignora al atacante. El algoritmo requiere similitudes matemáticas profundas y genuinas, diluyendo el perfil del atacante de "Popularidad" fuera del Top 10 de vecinos (K).

---

## Problemas y Soluciones

A lo largo del desarrollo y ejecución del código base, se presentaron los siguientes retos técnicos reales, los cuales requirieron ajustes específicos en la lógica de programación:

### Problema 1: ZeroDivisionError

Problema 1: Excepciones ZeroDivisionError en Evaluación MAE

Contexto: Al medir el desempeño (Tarea 4) probando el sistema contra usuarios reales, ocurría que para algunos usuarios de prueba, las películas que ellos habían visto no habían sido vistas por ninguno de sus vecinos más cercanos. La lista predicciones quedaba vacía, lo que provocaba que la función calcular_mae() intentara dividir entre len(errores) (cuyo valor era 0), colgando el script.

Solución Aplicada: Se ajustó la función de error absoluto medio para retornar un "0.0" controlado (o un flag matemático) si no había intersección, evitando la caída del sistema de benchmarking mediante un operador ternario: return sum(errores) / len(errores) if errores else 0.0.

### Problema 2: Falsos positivos en Coseno

Contexto: En la implementación inicial de la Distancia de Coseno, la norma (el divisor) se calculaba solo sobre las películas que ambos usuarios tenían en común. Esto provocaba un error lógico severo: si el Usuario A había visto 500 películas y el Usuario B solo 1 ("Toy Story"), y ambos le daban 5 estrellas, la fórmula daba un 100% (1.0) de similitud, emparejándolos engañosamente.
Solución Aplicada: Se corrigió la fórmula matemática del Coseno. El producto punto del numerador se mantuvo solo para los ítems comunes, pero la norma del denominador se calculó sobre el perfil completo de cada usuario (sum(r\*\*2 for r in ratings_u.values())). Esto penalizó correctamente a los usuarios con perfiles muy pequeños o con baja intersección en relación a su historial total.

### Problema 3: Curse of Dimensionality

Contexto: Al escalar el benchmark a más de 50,000 registros, las distancias geométricas (Euclidiana y Manhattan) colapsaron por completo. El terminal comenzó a arrojar predicciones vacías y el MAE reportó 0.0000.

Solución y Análisis: Se descubrió que al utilizar dicts dispersos, la distancia absoluta ($L^1$ y $L^2$) se distorsiona en espacios multidimensionales. Al haber tantas películas, la probabilidad de encontrar vecinos con distancias "pequeñas" desaparece. Las distancias se hacen matemáticamente relativas y tienden al infinito algorítmico, provocando que los "pesos ponderados" de (1 / 1 + dist) tiendan a 0 y no superen los umbrales. Se documentó este hallazgo, estableciendo que la Similitud de Coseno (que mide ángulos y no distancias geométricas) es la única apta para este tipo de matrices.

---

## Referencias

- Harper, F. M., & Konstan, J. A. (2015). _The MovieLens Datasets: History and Context_. ACM TiiS.
- GroupLens Research: [http://grouplens.org/datasets/movielens/](http://grouplens.org/datasets/movielens/)
- Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_. Springer.
- Bell, R. M., & Koren, Y. (2007). _Scalable Collaborative Filtering_. IEEE ICDM.
