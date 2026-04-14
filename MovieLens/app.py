import pandas as pd
import math
import random
import time
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. MÉTRICAS DE DISTANCIA / SIMILITUD
# ==========================================

def similitud_coseno(calificaciones_u: dict, calificaciones_v: dict) -> float:
    comunes = set(calificaciones_u.keys()) & set(calificaciones_v.keys())
    if not comunes:
        return 0.0
    producto_punto = sum(calificaciones_u[m] * calificaciones_v[m] for m in comunes)
    norma_u = math.sqrt(sum(r**2 for r in calificaciones_u.values()))
    norma_v = math.sqrt(sum(r**2 for r in calificaciones_v.values()))
    if norma_u == 0.0 or norma_v == 0.0:
        return 0.0
    return producto_punto / (norma_u * norma_v)

def distancia_euclidiana(calificaciones_u: dict, calificaciones_v: dict) -> float:
    comunes = set(calificaciones_u.keys()) & set(calificaciones_v.keys())
    if not comunes:
        return float('inf')
    suma_cuadrados = sum((calificaciones_u[m] - calificaciones_v[m]) ** 2 for m in comunes)
    return math.sqrt(suma_cuadrados)

def distancia_manhattan(calificaciones_u: dict, calificaciones_v: dict) -> float:
    comunes = set(calificaciones_u.keys()) & set(calificaciones_v.keys())
    if not comunes:
        return float('inf')
    return sum(abs(calificaciones_u[m] - calificaciones_v[m]) for m in comunes)

# ==========================================
# 2. ALGORITMO K-NN (TAREA 1)
# ==========================================

def obtener_knn(usuario_objetivo, calificaciones_usuarios, k=10, funcion_distancia=similitud_coseno, es_similitud=True):
    if usuario_objetivo not in calificaciones_usuarios:
        return []
    preferencias_objetivo = calificaciones_usuarios[usuario_objetivo]
    resultados = []
    for uid, prefs in calificaciones_usuarios.items():
        if uid == usuario_objetivo:
            continue
        puntaje = funcion_distancia(preferencias_objetivo, prefs)
        if es_similitud and puntaje > 0:
            resultados.append((uid, puntaje))
        elif not es_similitud and puntaje != float('inf'):
            resultados.append((uid, puntaje))
    resultados.sort(key=lambda x: x[1], reverse=es_similitud)
    return resultados[:k]

# ==========================================
# 3. ALGORITMO DE RECOMENDACIÓN (TAREA 2)
# ==========================================

def recomendar(target_uid, user_ratings, k=10, umbral=3.0):
    vecinos = obtener_knn(target_uid, user_ratings, k=k, funcion_distancia=similitud_coseno, es_similitud=True)
    if not vecinos:
        return []
    vistas = set(user_ratings[target_uid].keys())
    acumuladores = {}
    for v_id, sim in vecinos:
        v_prefs = user_ratings[v_id]
        for mid, rating in v_prefs.items():
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
    predicciones.sort(key=lambda x: x[1], reverse=True)
    return predicciones

# ==========================================
# 4. INFLUENCIA Y USUARIOS (TAREA 3)
# ==========================================

def crear_usuarios_batch(user_ratings, cantidad=5, num_ratings=20):
    todas_mids = list(set().union(*(d.keys() for d in user_ratings.values())))
    if not todas_mids:
        return []
    max_id = max(user_ratings.keys())
    nuevos_ids = []
    for i in range(1, cantidad + 1):
        new_id = max_id + i
        mids_sample = random.sample(todas_mids, min(num_ratings, len(todas_mids)))
        user_ratings[new_id] = {mid: round(random.uniform(1.0, 5.0), 1) for mid in mids_sample}
        nuevos_ids.append(new_id)
    return nuevos_ids

def crear_influencer(user_ratings, influencer_id=9999, top_n=100):
    conteo, sumas = {}, {}
    for prefs in user_ratings.values():
        for mid, r in prefs.items():
            conteo[mid] = conteo.get(mid, 0) + 1
            sumas[mid] = sumas.get(mid, 0) + r
    populares = sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:top_n]
    perfil_influencer = {mid: round(sumas[mid] / conteo[mid], 2) for mid, _ in populares}
    user_ratings[influencer_id] = perfil_influencer
    return perfil_influencer

# ==========================================
# 5. COMPLEJIDAD Y RENDIMIENTO (TAREA 4)
# ==========================================

def calcular_mae(predicciones, real_ratings_usuario):
    errores = []
    for mid, pred in predicciones:
        if mid in real_ratings_usuario:
            errores.append(abs(pred - real_ratings_usuario[mid]))
    return sum(errores) / len(errores) if errores else 0.0

def ejecutar_experimento_completo(path_csv, target_uid, cantidades):
    distancias = [
        ('Coseno',     similitud_coseno,    True),
        ('Euclidiana', distancia_euclidiana, False),
        ('Manhattan',  distancia_manhattan,  False)
    ]
    try:
        df_full = pd.read_csv(path_csv)
        max_rows = len(df_full)
    except FileNotFoundError:
        return None, f"Archivo no encontrado: {path_csv}"

    filas = []
    for n in cantidades:
        n = min(n, max_rows)
        start_load = time.time()
        df_temp = pd.read_csv(path_csv, nrows=n)
        data_test = {}
        for row in df_temp.itertuples(index=False):
            uid, mid, r = int(row.userId), int(row.movieId), float(row.rating)
            if uid not in data_test:
                data_test[uid] = {}
            data_test[uid][mid] = r
        time_subida = time.time() - start_load

        memoria = (sys.getsizeof(data_test) + sum(sys.getsizeof(v) for v in data_test.values())) / (1024 * 1024)

        uid_usado = target_uid if target_uid in data_test else list(data_test.keys())[0]
        target_prefs = data_test[uid_usado]

        for nombre_dist, func_dist, is_sim in distancias:
            start_dist = time.time()
            vecinos = obtener_knn(uid_usado, data_test, k=10, funcion_distancia=func_dist, es_similitud=is_sim)
            time_distancia = time.time() - start_dist

            start_reco = time.time()
            acum = {}
            for v_id, score in vecinos:
                if score == 0 or score == float('inf'):
                    continue
                for mid, r in data_test[v_id].items():
                    if mid in target_prefs:
                        if mid not in acum:
                            acum[mid] = [0.0, 0.0]
                        peso = score if is_sim else (1 / (1 + score))
                        acum[mid][0] += peso * r
                        acum[mid][1] += peso
            preds = [(m, (s[0] / s[1])) for m, s in acum.items() if s[1] > 0]
            time_reco = time.time() - start_reco

            precision = calcular_mae(preds, target_prefs)

            filas.append({
                "registros":    n,
                "distancia":    nombre_dist,
                "t_subida":     round(time_subida, 4),
                "t_distancia":  round(time_distancia, 4),
                "t_reco":       round(time_reco, 4),
                "ram_mb":       round(memoria, 2),
                "mae":          round(precision, 4)
            })
    return filas, None

# ==========================================
# ESTADO GLOBAL EN MEMORIA
# ==========================================

user_ratings_global = {}
csv_activo = None

def _get_metric_fn(nombre):
    return {
        'coseno':     (similitud_coseno,    True),
        'euclidiana': (distancia_euclidiana, False),
        'manhattan':  (distancia_manhattan,  False),
    }.get(nombre, (similitud_coseno, True))

def _load_csv(path):
    global user_ratings_global, csv_activo
    df = pd.read_csv(path)
    user_ratings_global = {}
    for row in df.itertuples(index=False):
        u, m, r = int(row.userId), int(row.movieId), float(row.rating)
        if u not in user_ratings_global:
            user_ratings_global[u] = {}
        user_ratings_global[u][m] = r
    csv_activo = path

# ==========================================
# ENDPOINTS FLASK
# ==========================================

@app.route('/api/cargar', methods=['POST'])
def api_cargar():
    data = request.json
    path = data.get('path', '')
    if not os.path.exists(path):
        return jsonify({"error": f"Archivo no encontrado: {path}"}), 404
    try:
        _load_csv(path)
        return jsonify({
            "usuarios": len(user_ratings_global),
            "ratings":  sum(len(v) for v in user_ratings_global.values()),
            "path":     path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/knn', methods=['POST'])
def api_knn():
    data = request.json
    uid    = int(data.get('usuario_id'))
    k      = int(data.get('k', 10))
    metrica = data.get('metrica', 'coseno')

    if not user_ratings_global:
        return jsonify({"error": "No hay datos cargados"}), 400
    if uid not in user_ratings_global:
        return jsonify({"error": f"Usuario {uid} no encontrado"}), 404

    fn, es_sim = _get_metric_fn(metrica)
    t0 = time.time()
    vecinos = obtener_knn(uid, user_ratings_global, k=k, funcion_distancia=fn, es_similitud=es_sim)
    elapsed = round((time.time() - t0) * 1000, 2)

    target_movies = set(user_ratings_global[uid].keys())
    result = []
    for vid, score in vecinos:
        v_movies = set(user_ratings_global[vid].keys())
        overlap = len(target_movies & v_movies)
        result.append({
            "usuario_id": vid,
            "puntaje":    round(score, 6),
            "peliculas":  len(v_movies),
            "overlap":    overlap,
            "overlap_pct": round(overlap / max(len(target_movies), 1) * 100, 1)
        })

    return jsonify({
        "usuario_id":     uid,
        "k":              k,
        "metrica":        metrica,
        "tiempo_ms":      elapsed,
        "peliculas_usuario": len(user_ratings_global[uid]),
        "vecinos":        result
    })


@app.route('/api/recomendar', methods=['POST'])
def api_recomendar():
    data    = request.json
    uid     = int(data.get('usuario_id'))
    k       = int(data.get('k', 10))
    umbral  = float(data.get('umbral', 3.0))
    top_n   = int(data.get('top_n', 10))

    if not user_ratings_global:
        return jsonify({"error": "No hay datos cargados"}), 400
    if uid not in user_ratings_global:
        return jsonify({"error": f"Usuario {uid} no encontrado"}), 404

    vecinos = obtener_knn(uid, user_ratings_global, k=k,
                          funcion_distancia=similitud_coseno, es_similitud=True)
    predicciones = recomendar(uid, user_ratings_global, k=k, umbral=umbral)

    return jsonify({
        "usuario_id":    uid,
        "vecinos_usados": len(vecinos),
        "candidatas":    len(predicciones),
        "recomendaciones": [
            {"rank": i + 1, "movie_id": mid, "prediccion": round(pred, 4)}
            for i, (mid, pred) in enumerate(predicciones[:top_n])
        ]
    })


@app.route('/api/batch', methods=['POST'])
def api_batch():
    data       = request.json
    cantidad   = int(data.get('cantidad', 5))
    num_ratings = int(data.get('num_ratings', 20))

    if not user_ratings_global:
        return jsonify({"error": "No hay datos cargados"}), 400

    nuevos_ids = crear_usuarios_batch(user_ratings_global, cantidad=cantidad, num_ratings=num_ratings)
    return jsonify({
        "nuevos_ids":     nuevos_ids,
        "total_usuarios": len(user_ratings_global)
    })


@app.route('/api/influencer', methods=['POST'])
def api_influencer():
    data         = request.json
    influencer_id = int(data.get('influencer_id', 9999))
    top_n        = int(data.get('top_n', 100))

    if not user_ratings_global:
        return jsonify({"error": "No hay datos cargados"}), 400

    perfil = crear_influencer(user_ratings_global, influencer_id=influencer_id, top_n=top_n)
    avg_rating = sum(perfil.values()) / len(perfil) if perfil else 0

    return jsonify({
        "influencer_id":  influencer_id,
        "num_peliculas":  len(perfil),
        "rating_promedio": round(avg_rating, 4),
        "total_usuarios": len(user_ratings_global)
    })


@app.route('/api/experimento', methods=['POST'])
def api_experimento():
    data       = request.json
    path       = data.get('path', csv_activo or '')
    target_uid = int(data.get('usuario_id', 1))
    cantidades = [int(c) for c in data.get('cantidades', [100, 500, 1000])]

    if not os.path.exists(path):
        return jsonify({"error": f"Archivo no encontrado: {path}"}), 404

    filas, error = ejecutar_experimento_completo(path, target_uid, cantidades)
    if error:
        return jsonify({"error": error}), 500

    return jsonify({"resultados": filas})


@app.route('/api/analisis', methods=['GET'])
def api_analisis():
    if not user_ratings_global:
        return jsonify({"error": "No hay datos cargados"}), 400

    todos_ratings = [r for prefs in user_ratings_global.values() for r in prefs.values()]
    peliculas     = set(m for prefs in user_ratings_global.values() for m in prefs.keys())
    ratings_por_usuario = {uid: len(prefs) for uid, prefs in user_ratings_global.items()}

    n  = len(todos_ratings)
    mu = sum(todos_ratings) / n if n else 0
    varianza = sum((r - mu) ** 2 for r in todos_ratings) / n if n else 0
    std = math.sqrt(varianza)

    sorted_users = sorted(ratings_por_usuario.items(), key=lambda x: x[1], reverse=True)
    top_usuarios = [{"usuario_id": uid, "ratings": cnt} for uid, cnt in sorted_users[:10]]

    buckets = [0] * 10
    for r in todos_ratings:
        idx = min(int(round(r * 2)) - 1, 9)
        if idx >= 0:
            buckets[idx] += 1

    num_u = len(user_ratings_global)
    num_m = len(peliculas)
    sparsity = (1 - n / (num_u * num_m)) * 100 if num_u and num_m else 100

    return jsonify({
        "num_usuarios":         num_u,
        "num_peliculas":        num_m,
        "num_ratings":          n,
        "rating_promedio":      round(mu, 4),
        "rating_std":           round(std, 4),
        "rating_min":           min(todos_ratings) if todos_ratings else 0,
        "rating_max":           max(todos_ratings) if todos_ratings else 0,
        "sparsity_pct":         round(sparsity, 4),
        "avg_ratings_usuario":  round(n / num_u, 2) if num_u else 0,
        "distribucion_ratings": buckets,
        "top_usuarios":         top_usuarios,
        "csv_activo":           csv_activo or "ninguno"
    })


@app.route('/api/estado', methods=['GET'])
def api_estado():
    return jsonify({
        "cargado":    bool(user_ratings_global),
        "usuarios":   len(user_ratings_global),
        "csv_activo": csv_activo or "ninguno"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)