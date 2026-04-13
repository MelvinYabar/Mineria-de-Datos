import math
import random
import time
import sys
import pandas as pd


# ─────────────────────────────────────────
# CREAR USUARIOS NUEVOS
# Se asume que user_ratings ya está cargado
# ─────────────────────────────────────────
ratings = pd.read_csv('data/ratings.csv')   # userId, movieId, rating, timestamp

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

def crear_usuario_aleatorio(user_id: int, num_ratings: int = 30) -> dict:
    """
    Crea un usuario nuevo con ratings aleatorios
    sobre películas existentes del dataset.
    """
    peliculas_disponibles = set()
    for rv in user_ratings.values():
        peliculas_disponibles.update(rv.keys())

    peliculas_sample = random.sample(list(peliculas_disponibles), num_ratings)

    nuevo = {movie_id: round(random.uniform(1.0, 5.0), 1)
             for movie_id in peliculas_sample}

    user_ratings[user_id] = nuevo
    print(f"  Usuario {user_id} creado con {num_ratings} ratings.")
    return nuevo


def crear_usuarios_batch(cantidad: int = 5, ratings_por_usuario: int = 30):
    """
    Crea múltiples usuarios nuevos con IDs consecutivos
    desde el máximo ID existente.
    """
    max_id    = max(user_ratings.keys())
    nuevos_ids = []

    print(f"\n[ Creando {cantidad} usuarios nuevos ]\n")

    for i in range(cantidad):
        nuevo_id = max_id + i + 1
        crear_usuario_aleatorio(nuevo_id, ratings_por_usuario)
        nuevos_ids.append(nuevo_id)

    print(f"\n  IDs generados: {nuevos_ids}")
    return nuevos_ids
# ─────────────────────────────────────────
# MÉTRICAS DE DISTANCIA 
# ─────────────────────────────────────────

def distancia_euclidiana(ratings_u: dict, ratings_v: dict) -> float:
    common = set(ratings_u.keys()) & set(ratings_v.keys())
    if not common:
        return float('inf')
    return math.sqrt(sum((ratings_u[m] - ratings_v[m]) ** 2 for m in common))


def pearson_correlation(ratings_u: dict, ratings_v: dict) -> float:
    common = set(ratings_u.keys()) & set(ratings_v.keys())
    if len(common) < 2:
        return 0.0

    n      = len(common)
    mean_u = sum(ratings_u[m] for m in common) / n
    mean_v = sum(ratings_v[m] for m in common) / n

    num    = sum((ratings_u[m] - mean_u) * (ratings_v[m] - mean_v) for m in common)
    den_u  = math.sqrt(sum((ratings_u[m] - mean_u) ** 2 for m in common))
    den_v  = math.sqrt(sum((ratings_v[m] - mean_v) ** 2 for m in common))

    if den_u == 0.0 or den_v == 0.0:
        return 0.0

    return num / (den_u * den_v)


def cosine_similarity(ratings_u: dict, ratings_v: dict) -> float:
    common = set(ratings_u.keys()) & set(ratings_v.keys())
    if not common:
        return 0.0

    dot    = sum(ratings_u[m] * ratings_v[m] for m in common)
    norm_u = math.sqrt(sum(ratings_u[m] ** 2 for m in common))
    norm_v = math.sqrt(sum(ratings_v[m] ** 2 for m in common))

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    return dot / (norm_u * norm_v)


# ─────────────────────────────────────────
# CREAR INFLUENCER
# Estrategia: tomar las películas más
# calificadas del dataset y asignarles
# el rating promedio global → maximiza
# películas en común con todos los usuarios
# ─────────────────────────────────────────

def crear_influencer(user_id: int = 9999) -> dict:
    """
    Construye un usuario influencer cuyo perfil maximiza
    la similitud con el mayor número de usuarios posible.

    Estrategia:
      - Calificar las N películas más populares (más calificadas)
      - Asignar a cada película su rating promedio global
        → hace que la similitud coseno y pearson sean altas
          con casi todos los usuarios
    """
    # Contar cuántos usuarios calificaron cada película
    conteo   = {}
    suma     = {}

    for rv in user_ratings.values():
        for mid, r in rv.items():
            conteo[mid] = conteo.get(mid, 0) + 1
            suma[mid]   = suma.get(mid, 0.0) + r

    # Ordenar películas por popularidad (insertion sort sobre items)
    items = list(conteo.items())
    for i in range(1, len(items)):
        key = items[i]
        j   = i - 1
        while j >= 0 and items[j][1] < key[1]:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = key

    # Tomar top-200 películas más populares
    top_peliculas = [mid for mid, _ in items[:200]]

    # Rating = promedio global de cada película
    influencer = {mid: round(suma[mid] / conteo[mid], 2)
                  for mid in top_peliculas}

    user_ratings[user_id] = influencer
    print(f"  Influencer {user_id} creado con {len(influencer)} ratings estratégicos.")
    return influencer


def analizar_influencer(influencer_id: int = 9999, muestra: int = 50):
    """
    Compara las 3 métricas para medir qué tan 'cercano'
    es el influencer con respecto a los demás usuarios.
    """
    influencer = user_ratings[influencer_id]
    usuarios_muestra = [uid for uid in list(user_ratings.keys())[:muestra]
                        if uid != influencer_id]

    resultados = []

    for uid in usuarios_muestra:
        rv  = user_ratings[uid]
        cos = cosine_similarity(influencer, rv)
        pea = pearson_correlation(influencer, rv)
        euc = distancia_euclidiana(influencer, rv)
        resultados.append((uid, cos, pea, euc))

    # Promedios
    avg_cos = sum(r[1] for r in resultados) / len(resultados)
    avg_pea = sum(r[2] for r in resultados) / len(resultados)
    avg_euc = sum(r[3] for r in resultados if r[3] != float('inf')) / len(resultados)

    print("\n" + "=" * 65)
    print(f"  ANÁLISIS INFLUENCER  |  ID: {influencer_id}  |  Muestra: {muestra} usuarios")
    print("=" * 65)
    print(f"  {'Métrica':<25} {'Promedio':<15} {'Interpretación'}")
    print("-" * 65)
    print(f"  {'Similitud Coseno':<25} {avg_cos:<15.4f} {'↑ más alto = más influencia'}")
    print(f"  {'Correlación Pearson':<25} {avg_pea:<15.4f} {'↑ más alto = más influencia'}")
    print(f"  {'Distancia Euclidiana':<25} {avg_euc:<15.4f} {'↓ más bajo = más influencia'}")
    print("=" * 65)

    # Veredicto
    metricas = {
        'Coseno'    : avg_cos,
        'Pearson'   : avg_pea,
        'Euclidiana': 1 / (1 + avg_euc)   # normalizar para comparar
    }
    mejor = max(metricas, key=metricas.get)
    print(f"\n  ✔ Métrica más favorable para el influencer: {mejor}")
    print(f"    Usar esta métrica maximiza su aparición como vecino cercano.")
    print("=" * 65)

    return resultados


# ─────────────────────────────────────────
# MÉTRICAS DE COMPUTACIÓN
# Tiempo, operaciones y memoria — sin libs
# ─────────────────────────────────────────

contador_operaciones = {"comparaciones": 0, "multiplicaciones": 0, "sumas": 0}

def cosine_similarity_instrumented(ratings_u: dict, ratings_v: dict) -> float:
    """Versión instrumentada que cuenta operaciones."""
    common = set(ratings_u.keys()) & set(ratings_v.keys())

    if not common:
        return 0.0

    dot, norm_u, norm_v = 0.0, 0.0, 0.0

    for m in common:
        contador_operaciones["multiplicaciones"] += 1
        contador_operaciones["sumas"]            += 1
        dot    += ratings_u[m] * ratings_v[m]
        norm_u += ratings_u[m] ** 2
        norm_v += ratings_v[m] ** 2

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_u) * math.sqrt(norm_v))


def medir_knn(target_user_id: int, k: int = 10):
    """
    Ejecuta K-NN midiendo:
      - Tiempo de ejecución (segundos)
      - Número de operaciones
      - Memoria estimada del dict user_ratings
    """
    global contador_operaciones
    contador_operaciones = {"comparaciones": 0, "multiplicaciones": 0, "sumas": 0}

    # ── Memoria estimada ──
    bytes_total = 0
    for uid, rv in user_ratings.items():
        bytes_total += sys.getsizeof(uid)
        bytes_total += sys.getsizeof(rv)
        for mid, r in rv.items():
            bytes_total += sys.getsizeof(mid) + sys.getsizeof(r)

    # ── Tiempo ──
    t_inicio = time.time()

    target      = user_ratings[target_user_id]
    similarities = []

    for uid, rv in user_ratings.items():
        if uid == target_user_id:
            continue
        contador_operaciones["comparaciones"] += 1
        sim = cosine_similarity_instrumented(target, rv)
        similarities.append((uid, sim))

    # Ordenar
    for i in range(1, len(similarities)):
        key = similarities[i]
        j   = i - 1
        while j >= 0 and similarities[j][1] < key[1]:
            contador_operaciones["comparaciones"] += 1
            similarities[j + 1] = similarities[j]
            j -= 1
        similarities[j + 1] = key

    t_fin = time.time()

    vecinos   = similarities[:k]
    t_total   = t_fin - t_inicio
    total_ops = sum(contador_operaciones.values())

    # ── Salida ──
    print("\n" + "=" * 55)
    print(f"  MÉTRICAS DE COMPUTACIÓN  |  Usuario: {target_user_id}")
    print("=" * 55)
    print(f"  {'Tiempo de ejecución':<30} {t_total:.4f} seg")
    print(f"  {'Usuarios comparados':<30} {len(similarities)}")
    print(f"  {'Operaciones totales':<30} {total_ops:,}")
    print(f"    {'· Comparaciones':<28} {contador_operaciones['comparaciones']:,}")
    print(f"    {'· Multiplicaciones':<28} {contador_operaciones['multiplicaciones']:,}")
    print(f"    {'· Sumas':<28} {contador_operaciones['sumas']:,}")
    print(f"  {'Memoria estimada (ratings)':<30} {bytes_total / 1024:.2f} KB")
    print(f"  {'Complejidad temporal':<30} O(n × m)  n=usuarios, m=películas")
    print(f"  {'Complejidad espacial':<30} O(n × m)  matriz dispersa")
    print("=" * 55)

    return vecinos


# ─────────────────────────────────────────
# EJECUCIÓN COMPLETA TAREA 3
# ─────────────────────────────────────────
if __name__ == "__main__":

    # 1. Crear usuarios nuevos
    crear_usuarios_batch(cantidad=5, ratings_por_usuario=30)

    # 2. Crear y analizar influencer
    crear_influencer(user_id=9999)
    analizar_influencer(influencer_id=9999, muestra=50)

    # 3.5 Métricas de computación
    medir_knn(target_user_id=1, k=10)