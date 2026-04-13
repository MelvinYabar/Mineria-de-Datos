import pandas as pd
import math
from influencer import crear_usuarios_batch, crear_influencer, analizar_influencer, medir_knn

# ─────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────
ratings = pd.read_csv('data/example.csv')   # userId, movieId, rating, timestamp

rating_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

user_ratings = {}
for user_id, row in rating_matrix.iterrows():
    user_ratings[user_id] = {
        movie_id: r
        for movie_id, r in row.items()
        if not math.isnan(r)
    }

print(f"Usuarios cargados: {len(user_ratings)}")


# ─────────────────────────────────────────
# SIMILITUD COSENO (implementada a mano)
# ─────────────────────────────────────────
def cosine_similarity(ratings_u: dict, ratings_v: dict) -> float:
    common = set(ratings_u.keys()) & set(ratings_v.keys())

    if not common:
        return 0.0

    dot_product = sum(ratings_u[m] * ratings_v[m] for m in common)
    norm_u      = math.sqrt(sum(ratings_u[m] ** 2 for m in common))
    norm_v      = math.sqrt(sum(ratings_v[m] ** 2 for m in common))

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    return dot_product / (norm_u * norm_v)


# ─────────────────────────────────────────
# ORDENAMIENTO (insertion sort manual)
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
# TAREA 1 — K VECINOS MÁS CERCANOS
# ─────────────────────────────────────────
def knn_users(target_user_id: int, k: int = 10) -> list:
    if target_user_id not in user_ratings:
        raise ValueError(f"Usuario {target_user_id} no encontrado en el dataset.")

    target      = user_ratings[target_user_id]
    similarities = []

    for user_id, ratings_v in user_ratings.items():
        if user_id == target_user_id:
            continue
        sim = cosine_similarity(target, ratings_v)
        similarities.append((user_id, sim))

    similarities = insertion_sort_desc(similarities)

    # ── Salida K vecinos ──
    target_movies = set(user_ratings[target_user_id].keys())

    print("=" * 55)
    print(f"  K-NN  |  Usuario objetivo: {target_user_id}  |  K = {k}")
    print("=" * 55)
    print(f"{'Rank':<6} {'UserID':<10} {'Similitud':<14} {'En común':<10} {'Barra'}")
    print("-" * 55)

    for rank, (uid, sim) in enumerate(similarities[:k], start=1):
        common = len(target_movies & set(user_ratings[uid].keys()))
        print(f"{rank:<6} {uid:<10} {sim:<14.4f} {common:<10}")

    print("-" * 55)
    print(f"  Vecino más cercano : Usuario {similarities[0][0]}  (sim={similarities[0][1]:.4f})")
    #print(f"  Vecino más lejano  : Usuario {similarities[k-1][0]}  (sim={similarities[k-1][1]:.4f})")
    print("=" * 55)

    return similarities[:k]


# ─────────────────────────────────────────
# TAREA 2 — SISTEMA DE RECOMENDACIÓN
# ─────────────────────────────────────────

UMBRAL = 3.0

def peliculas_no_vistas(target_user_id: int) -> set:
    todas   = set()
    for ratings_v in user_ratings.values():
        todas.update(ratings_v.keys())

    vistas   = set(user_ratings[target_user_id].keys())
    no_vistas = todas - vistas

    print(f"  Películas vistas    : {len(vistas)}")
    print(f"  Películas no vistas : {len(no_vistas)}")
    print(f"  Total en dataset    : {len(todas)}")

    return no_vistas


def predecir_rating(movie_id: int, vecinos: list) -> float:
    numerador   = 0.0
    denominador = 0.0

    for (uid, sim) in vecinos:
        if movie_id in user_ratings[uid]:
            numerador   += sim * user_ratings[uid][movie_id]
            denominador += abs(sim)

    if denominador == 0.0:
        return None

    return numerador / denominador


def recomendar(target_user_id: int, k: int = 10, umbral: float = UMBRAL, top_n: int = 10):
    print("=" * 60)
    print(f"  RECOMENDACIONES  |  Usuario: {target_user_id}  |  Umbral: {umbral}  |  K: {k}")
    print("=" * 60)

    # Paso 1 — películas no vistas
    print("\n[ Paso 1 ] Películas no vistas por el usuario:")
    no_vistas = peliculas_no_vistas(target_user_id)

    # K-NN
    print(f"\n[ K-NN   ] Calculando {k} vecinos más cercanos...")
    vecinos = knn_users(target_user_id, k=k)

    # Paso 2 y 3 — predecir y filtrar
    print(f"\n[ Paso 2 ] Prediciendo ratings  →  [ Paso 3 ] Filtrando por umbral > {umbral}\n")

    predicciones = []
    for movie_id in no_vistas:
        pred = predecir_rating(movie_id, vecinos)
        if pred is None:
            continue
        if pred > umbral:
            predicciones.append((movie_id, pred))

    predicciones = insertion_sort_desc(predicciones)

    if not predicciones:
        print(f"  No se encontraron películas con predicción > {umbral}")
        return []

    try:
        movies_df = pd.read_csv('movies.csv')
        titles    = dict(zip(movies_df['movieId'], movies_df['title']))
    except FileNotFoundError:
        titles = {}

    top = predicciones[:top_n]

    print(f"{'Rank':<5} {'MovieID':<10} {'Rating pred.':<15} {'Estrellas':<12} {'Título'}")
    print("-" * 60)

    for rank, (mid, pred) in enumerate(top, start=1):
        titulo = titles.get(mid, "Título no disponible")
        barra  = "★" * int(pred)
        print(f"{rank:<5} {mid:<10} {pred:<15.4f} {barra:<12} {titulo}")

    print("-" * 60)
    mejor_id, mejor_pred = top[0]
    mejor_titulo = titles.get(mejor_id, f"Película {mejor_id}")
    print(f"\n  ✔ Recomendación principal para Usuario {target_user_id}:")
    print(f"    '{mejor_titulo}'  →  Rating predicho: {mejor_pred:.4f}")
    print("=" * 60)

    return top


# ─────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────
if __name__ == "__main__":
    recomendar(target_user_id=8, k=10, umbral=3.0, top_n=5)

        # 1. Crear usuarios nuevos
    crear_usuarios_batch(cantidad=5, ratings_por_usuario=7)

    # 2. Crear y analizar influencer
    crear_influencer(user_id=9999)
    analizar_influencer(influencer_id=9999, muestra=50)

    # 3.5 Métricas de computación
    medir_knn(target_user_id=1, k=10)