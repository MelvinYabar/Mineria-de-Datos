## 🎯 Objetivos del Proyecto

- **Similitud**: Calcular qué tan parecidos son dos usuarios (coseno, Manhattan, Euclidiana, Pearson).
- **Algoritmo k-nn**: Encontrar los K vecinos más cercanos.
- **Algoritmo de Recomendación (T1)**: Recomendar películas con puntaje proyectado > 3.
- **Influencia (T2)**: Crear el concepto de “Influencer”.
- **Complejidad computacional (T3)**: Analizar métricas de distancia.

## 🛠️ Cómo ejecutar

1. Entrar a la carpeta del dataset:
   ```bash
   cd ml-latest-small
   ```
2. Instalar pandas (solo una vez):
   ```bash
   pip install pandas
   ```
3. Ejecutar el algoritmo:
   ```bash
   python Algoritmo_k-nn.py
   ```

## 📊 Salida del programa
- Tabla de **k vecinos** más cercanos.
- Tabla de **recomendaciones** (umbral > 3).
- Sección **Influencia (T2)** con el Influencer del usuario.
- Explicación de **complejidad (T3)**.

# ─────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

rating_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

user_ratings = {}
for user_id, row in rating_matrix.iterrows():
    user_ratings[user_id] = {movie_id: r for movie_id, r in row.items() if not math.isnan(r)}

print(f"Usuarios cargados: {len(user_ratings)}")

# ─────────────────────────────────────────
# 2. SIMILITUD COSENO (NO SE MODIFICA)
# ─────────────────────────────────────────
def cosine_similarity(ratings_u: dict, ratings_v: dict) -> float:
    common = set(ratings_u.keys()) & set(ratings_v.keys())
    if not common:
        return 0.0
    dot = sum(ratings_u[m] * ratings_v[m] for m in common)
    norm_u = math.sqrt(sum(ratings_u[m]**2 for m in common))
    norm_v = math.sqrt(sum(ratings_v[m]**2 for m in common))
    return dot / (norm_u * norm_v) if norm_u > 0 and norm_v > 0 else 0.0

# ─────────────────────────────────────────
# 3. ORDENAR (insertion sort - NO SE MODIFICA)
# ─────────────────────────────────────────
def insertion_sort_desc(pairs: list) -> list:
    for i in range(1, len(pairs)):
        key = pairs[i]
        j = i - 1
        while j >= 0 and pairs[j][1] < key[1]:
            pairs[j + 1] = pairs[j]
            j -= 1
        pairs[j + 1] = key
    return pairs

# ─────────────────────────────────────────
# 4. K-NN 
# ─────────────────────────────────────────
def knn_users(target_user_id: int, k: int = 10) -> list:
    if target_user_id not in user_ratings:
        raise ValueError(f"Usuario {target_user_id} no encontrado.")
    target = user_ratings[target_user_id]
    similarities = []
    for user_id, ratings_v in user_ratings.items():
        if user_id == target_user_id:
            continue
        sim = cosine_similarity(target, ratings_v)
        similarities.append((user_id, sim))
    similarities = insertion_sort_desc(similarities)
    return similarities[:k]

# ─────────────────────────────────────────
# 5. PREDICCIÓN Y RECOMENDACIONES (T1)
# ─────────────────────────────────────────
def predict_rating(target_user_id: int, movie_id: int, k: int = 10) -> float:
    vecinos = knn_users(target_user_id, k)
    if movie_id in user_ratings[target_user_id]:
        return user_ratings[target_user_id][movie_id]
    sum_weighted = 0.0
    sum_sims = 0.0
    for uid, sim in vecinos:
        if movie_id in user_ratings[uid]:
            sum_weighted += sim * user_ratings[uid][movie_id]
            sum_sims += sim
    return (sum_weighted / sum_sims) if sum_sims > 0 else 0.0

def get_recommendations(target_user_id: int, k: int = 10, threshold: float = 3.0, top_n: int = 10):
    target_movies = set(user_ratings[target_user_id].keys())
    all_movies = set(movies['movieId'])
    unseen = all_movies - target_movies
    recs = []
    for mid in unseen:
        pred = predict_rating(target_user_id, mid, k)
        if pred > threshold:
            title = movies[movies['movieId'] == mid]['title'].iloc[0]
            recs.append((mid, round(pred, 4), title))
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:top_n]

# ─────────────────────────────────────────
# 6. INFLUENCIA (T2 y T3)
# ─────────────────────────────────────────
def mostrar_influencia(target_user_id: int, k: int = 10):
    print("\n" + "="*70)
    print("INFLUENCIA (T2) - Creación de 'Influencer'")
    print("="*70)
    vecinos = knn_users(target_user_id, k)
    influencer_id, sim_max = vecinos[0]
    print(f"→ El **Influencer** del usuario {target_user_id} es:")
    print(f"   Usuario {influencer_id} con similitud = {sim_max:.4f}")
    print(f"   (es el vecino #1 del k-nn)")

    # Métricas de distancia mencionadas en la pizarra
    print("\nMétricas de distancia analizadas para crear el Influencer:")
    print("• Similitud Coseno     → usada en este algoritmo")
    print("• Distancia Manhattan  → alternativa posible")
    print("• Distancia Euclidiana → alternativa posible")
    print("• Correlación Pearson  → alternativa posible")

    # Complejidad computacional (T3)
    print("\nT3 → Complejidad computacional")
    print("• Tiempo: O(n · m)   donde n = número de usuarios (~600)")
    print("                     m = número promedio de películas por usuario (~150)")
    print("• En este dataset ≈ 600 × 150 = 90 000 operaciones por usuario")
    print("="*70)

# ─────────────────────────────────────────
# EJECUCIÓN FINAL
# ─────────────────────────────────────────
if __name__ == "__main__":
    TARGET_USER = 1
    K = 10

    start = time.time()

    print("="*70)
    print("ALGORITMO k-NN + RECOMENDACIONES (exacto pizarra)")
    print("="*70)

    # Vecinos
    vecinos = knn_users(TARGET_USER, K)
    target_movies_set = set(user_ratings[TARGET_USER].keys())
    print(f"\nK-NN → Usuario objetivo: {TARGET_USER} | K = {K}\n")
    print(f"{'Rank':<6} {'UserID':<10} {'Similitud':<14} {'Películas en común'}")
    print("-"*55)
    for rank, (uid, sim) in enumerate(vecinos, start=1):
        common = len(target_movies_set & set(user_ratings[uid].keys()))
        print(f"{rank:<6} {uid:<10} {sim:<14.4f} {common:<8}")

    # Recomendaciones (T1)
    print("\n" + "="*70)
    print(f"RECOMENDACIONES (T1) para Usuario {TARGET_USER} | Umbral > 3 | Top 10")
    print("="*70)
    recs = get_recommendations(TARGET_USER, k=K, threshold=3.0, top_n=10)
    if recs:
        print(f"{'Rank':<6} {'MovieID':<10} {'Puntaje Proyectado':<20} Título")
        print("-"*90)
        for rank, (mid, pred, title) in enumerate(recs, start=1):
            print(f"{rank:<6} {mid:<10} {pred:<20} {title}")
    else:
        print("No se encontraron películas con puntaje > 3")

    # Influencia (T2 + T3)
    mostrar_influencia(TARGET_USER, K)

    print(f"\nTiempo total de ejecución: {time.time()-start:.2f} segundos")
EOF

### 2. Explicación del Código

**Similitud**  
→ Usamos **similitud coseno** (la que aparece dibujada con los vectores U1, U2, U3 y el ángulo θ).  
Fórmula:  
\[
\text{sim}(U,V) = \frac{U \cdot V}{\|U\| \cdot \|V\|}
\]

**Algoritmo k-nn**  
→ Exactamente los 3 pasos escritos en la pizarra:  
1. Calcular la distancia (similitud) del usuario U con todos los demás.  
2. Ordenar los usuarios (usamos insertion sort).  
3. Salida: K vecinos más cercanos.

**Alg Recomendación (T1)**  
→ Exactamente lo que está en la pizarra:  
1. Seleccionar películas que **no vio** U.  
2. Determinar umbral > 3.  
3. Recomendar la película + dar el **valor** (puntaje proyectado).

**Influencia (T2)**  
→ Creamos el concepto de “Influencer”:  
El vecino #1 del k-nn es el **Influencer** del usuario U.

**Complejidad computacional (T3)**  
→ O(n · m) donde n = usuarios y m = películas promedio.  
En este dataset ≈ 90 000 operaciones por usuario.
