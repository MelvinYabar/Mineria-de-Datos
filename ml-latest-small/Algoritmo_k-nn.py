import pandas as pd
import math

# ─────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────
ratings = pd.read_csv('ratings.csv')   # userId, movieId, rating, timestamp
movies = pd.read_csv('movies.csv')
full_data = pd.merge(ratings, movies, on='movieId')

# Construir matriz usuario × película (sin librerías externas para el cálculo)
rating_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# Convertir a dict de dicts para trabajar sin numpy
# { userId: { movieId: rating, ... }, ... }
user_ratings = {}
for user_id, row in rating_matrix.iterrows():
    user_ratings[user_id] = {
        movie_id: r
        for movie_id, r in row.items()
        if not math.isnan(r)          # ignorar películas no calificadas
    }

print(f"Usuarios cargados: {len(user_ratings)}")


# ─────────────────────────────────────────
# 2. SIMILITUD COSENO (sin numpy ni scipy)
#    sim(U, V) = (U·V) / (||U|| * ||V||)
# ─────────────────────────────────────────
def cosine_similarity(ratings_u: dict, ratings_v: dict) -> float:
    """
    Calcula la similitud coseno entre dos usuarios.
    Solo considera películas calificadas por AMBOS usuarios.
    Retorna 0.0 si no comparten ninguna película.
    """
    # Películas en común
    common = set(ratings_u.keys()) & set(ratings_v.keys())

    if not common:
        return 0.0

    # Producto punto
    dot_product = sum(ratings_u[m] * ratings_v[m] for m in common)

    # Normas (sobre películas en común para evitar sesgo por cantidad)
    norm_u = math.sqrt(sum(ratings_u[m] ** 2 for m in common))
    norm_v = math.sqrt(sum(ratings_v[m] ** 2 for m in common))

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    return dot_product / (norm_u * norm_v)


# ─────────────────────────────────────────
# 3. ORDENAR (insertion sort manual, sin sorted())
# ─────────────────────────────────────────
def insertion_sort_desc(pairs: list) -> list:
    """
    Ordena lista de (user_id, similitud) de mayor a menor similitud.
    O(n²) — aceptable para n=600 usuarios.
    """
    for i in range(1, len(pairs)):
        key = pairs[i]
        j = i - 1
        while j >= 0 and pairs[j][1] < key[1]:   # comparar similitud
            pairs[j + 1] = pairs[j]
            j -= 1
        pairs[j + 1] = key
    return pairs


# ─────────────────────────────────────────
# 4. K-NN PRINCIPAL
# ─────────────────────────────────────────
def knn_users(target_user_id: int, k: int = 10) -> list:
    """
    Retorna los K vecinos más cercanos al usuario objetivo.

    Parámetros
    ----------
    target_user_id : int   ID del usuario de referencia (U)
    k              : int   Número de vecinos a retornar

    Retorna
    -------
    Lista de tuplas [(user_id, similitud), ...] ordenada desc, longitud K
    """
    if target_user_id not in user_ratings:
        raise ValueError(f"Usuario {target_user_id} no encontrado en el dataset.")

    target = user_ratings[target_user_id]
    similarities = []

    # Paso 1: calcular distancia con cada otro usuario
    for user_id, ratings_v in user_ratings.items():
        if user_id == target_user_id:
            continue                              # excluir al propio usuario
        sim = cosine_similarity(target, ratings_v)
        similarities.append((user_id, sim))

    # Paso 2: ordenar de mayor a menor similitud (manual)
    similarities = insertion_sort_desc(similarities)

    # Paso 3: retornar los K más cercanos
    return similarities[:k]


# ─────────────────────────────────────────
# 5. EJECUCIÓN Y SALIDA
# ─────────────────────────────────────────
if __name__ == "__main__":
    TARGET_USER = 1    # ← cambia este ID
    K = 10

    vecinos = knn_users(target_user_id=TARGET_USER, k=K)

    target_movies = set(user_ratings[TARGET_USER].keys())

    print("=" * 55)
    print(f"  K-NN  |  Usuario objetivo: {TARGET_USER}  |  K = {K}")
    print("=" * 55)
    print(f"{'Rank':<6} {'UserID':<10} {'Similitud':<14} {'Películas en común'}")
    print("-" * 55)

    for rank, (uid, sim) in enumerate(vecinos, start=1):
        common = len(target_movies & set(user_ratings[uid].keys()))
        print(f"{rank:<6} {uid:<10} {sim:<14.4f} {common:<8}")

    print("-" * 55)
    print(f"  Vecino más cercano : Usuario {vecinos[0][0]}  (sim={vecinos[0][1]:.4f})")
    print(f"  Vecino más lejano  : Usuario {vecinos[-1][0]}  (sim={vecinos[-1][1]:.4f})")
    print("=" * 55)
    