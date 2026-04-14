import pandas as pd
import math
import random
import time
import sys

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
        
        # Filtrar vecinos sin coincidencias
        if es_similitud and puntaje > 0:
            resultados.append((uid, puntaje))
        elif not es_similitud and puntaje != float('inf'):
            resultados.append((uid, puntaje))
            
    # Si es similitud, mayor es mejor. Si es distancia, menor es mejor.
    resultados.sort(key=lambda x: x[1], reverse=es_similitud)
    
    return resultados[:k]
# ==========================================
# 3. ALGORITMO DE RECOMENDACIÓN (TAREA 2)
# ==========================================

def recomendar(target_uid, user_ratings, k=10, umbral=3.0):
    print(f"\n[ Generando recomendaciones para Usuario {target_uid} (K={k}, Umbral={umbral}) ]")
    
    vecinos = obtener_knn(target_uid, user_ratings, k=k, funcion_distancia=similitud_coseno, es_similitud=True)
    if not vecinos:
        print("✘ No se encontraron vecinos con similitud positiva.")
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
    
    if not predicciones:
        print(f"✘ No hay películas que superen el umbral de {umbral}.")
        return []

    print(f"✔ Se encontraron {len(predicciones)} películas candidatas.")
    print(f"{'Rank':<6} {'MovieID':<10} {'Predicción':<12}")
    print("-" * 30)
    for rank, (mid, score) in enumerate(predicciones[:10], start=1):
        print(f"{rank:<6} {mid:<10} {score:.2f}")
        
    return predicciones

# ==========================================
# 4. INFLUENCIA Y USUARIOS (TAREA 3)
# ==========================================

def crear_usuarios_batch(user_ratings, cantidad=5, num_ratings=20):
    print(f"\n[ Creando {cantidad} usuarios aleatorios ]")
    todas_mids = list(set().union(*(d.keys() for d in user_ratings.values())))
    if not todas_mids: return []
    
    max_id = max(user_ratings.keys())
    nuevos_ids = []
    for i in range(1, cantidad + 1):
        new_id = max_id + i
        mids_sample = random.sample(todas_mids, min(num_ratings, len(todas_mids)))
        user_ratings[new_id] = {mid: round(random.uniform(1.0, 5.0), 1) for mid in mids_sample}
        nuevos_ids.append(new_id)
    print(f"✔ IDs generados: {nuevos_ids}")
    return nuevos_ids

def crear_influencer(user_ratings, influencer_id=9999):
    print(f"\n[ Creando perfil Influencer ID: {influencer_id} ]")
    conteo, sumas = {}, {}
    for prefs in user_ratings.values():
        for mid, r in prefs.items():
            conteo[mid] = conteo.get(mid, 0) + 1
            sumas[mid] = sumas.get(mid, 0) + r
            
    # Tomar las 100 más populares y asignarles el promedio global
    populares = sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:100]
    perfil_influencer = {mid: round(sumas[mid]/conteo[mid], 2) for mid, _ in populares}
    user_ratings[influencer_id] = perfil_influencer
    print(f"✔ Influencer creado con {len(perfil_influencer)} ratings de alta popularidad.")
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

def ejecutar_experimento_completo(path_csv, target_uid):
    # Asegúrate de que tu dataset tiene suficientes registros para estos cortes
    cantidades = [10000, 50000, 100000] 
    distancias = [
        ('Coseno', similitud_coseno, True),
        ('Euclidiana', distancia_euclidiana, False),
        ('Manhattan', distancia_manhattan, False)
    ]

    print(f"\n{'='*100}")
    print(f"{'TABLA DE ANÁLISIS DE RENDIMIENTO Y PRECISIÓN':^100}")
    print(f"{'='*100}\n")
    
    header = f"{'Registros':<10} | {'Distancia':<12} | {'T.Subida':<10} | {'T.Dist':<10} | {'T.Reco':<10} | {'RAM':<10} | {'MAE (Error)':<10}"
    print(header)
    print("-" * len(header))

    try:
        # Carga completa para saber el máximo disponible
        df_full = pd.read_csv(path_csv)
        max_rows = len(df_full)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {path_csv}.")
        return

    for n in cantidades:
        if n > max_rows: n = max_rows # Evitar error si el dataset es más pequeño que N
            
        # 1. Medir "SubirInformacion"
        start_load = time.time()
        df_temp = pd.read_csv(path_csv, nrows=n)
        data_test = {}
        for row in df_temp.itertuples(index=False):
            uid, mid, r = int(row.userId), int(row.movieId), float(row.rating)
            if uid not in data_test: data_test[uid] = {}
            data_test[uid][mid] = r
        time_subida = time.time() - start_load

        # 2. Medir "Capacidad Almacenamiento" (RAM)
        memoria = (sys.getsizeof(data_test) + sum(sys.getsizeof(v) for v in data_test.values())) / (1024*1024)

        if target_uid not in data_test:
            # Si el usuario objetivo no está en el corte actual, tomamos uno aleatorio
            target_uid = list(data_test.keys())[0]

        target_prefs = data_test[target_uid]

        for nombre_dist, func_dist, is_sim in distancias:
            # 3. Medir "TiempoDistancia"
            start_dist = time.time()
            vecinos = obtener_knn(target_uid, data_test, k=10, funcion_distancia=func_dist, es_similitud=is_sim)
            time_distancia = time.time() - start_dist

            # 4. Medir "Recomendacion"
            start_reco = time.time()
            acum = {}
            # Para medir MAE real, predecimos películas que el usuario SÍ vio.
            for v_id, score in vecinos:
                if score == 0 or score == float('inf'): continue
                for mid, r in data_test[v_id].items():
                    if mid in target_prefs: # Solo para calcular el error contra la realidad
                        if mid not in acum: acum[mid] = [0.0, 0.0]
                        # Peso: si es distancia, invertimos para que menor distancia = mayor peso
                        peso = score if is_sim else (1 / (1 + score))
                        acum[mid][0] += peso * r
                        acum[mid][1] += peso
            
            preds = [(m, (s[0]/s[1])) for m, s in acum.items() if s[1] > 0]
            time_reco = time.time() - start_reco

            # 5. Medir "Precision" (MAE)
            precision = calcular_mae(preds, target_prefs)

            # Imprimir Fila
            print(f"{n:<10} | {nombre_dist:<12} | {time_subida:<9.4f}s | {time_distancia:<9.4f}s | {time_reco:<9.4f}s | {memoria:<7.2f} MB | {precision:<10.4f}")

# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # IMPORTANTE: Reemplaza con la ruta real de tu archivo ratings.csv de MovieLens
    ruta_dataset = 'example.csv' 
    
    print("Iniciando sistema...")
    # Carga inicial completa para las tareas 1, 2 y 3
    try:
        df = pd.read_csv(ruta_dataset)
        user_ratings_global = {}
        for row in df.itertuples(index=False):
            u, m, r = int(row.userId), int(row.movieId), float(row.rating)
            if u not in user_ratings_global: user_ratings_global[u] = {}
            user_ratings_global[u][m] = r


        print(f"✔ Usuarios cargados: {len(user_ratings_global)}")

        # INPUTS
        usuario_objetivo = int(input("Ingrese el ID del usuario objetivo: ")) 
        k = int(input("Ingrese el valor de K para vecinos: "))
        cantidad = int(input("Ingrese la cantidad de usuarios a crear para el batch: "))
        num_ratings = int(input("Ingrese la cantidad de calificaciones por usuario: "))


        # Tareas 1 y 2
        recomendar(usuario_objetivo, user_ratings_global, k)
        nuevos_vecinos = obtener_knn(int(usuario_objetivo), user_ratings_global, k)
        for uid, sim in nuevos_vecinos:
            print(f"\nVecino: {uid} | Similitud: {sim:.4f}")
        
        # Tarea 3
        crear_usuarios_batch(user_ratings_global, cantidad, num_ratings)
        crear_influencer(user_ratings_global, influencer_id=8888)
        recomendar(usuario_objetivo, user_ratings_global, k) # Recomendar post-influencer
        nuevos_vecinos = obtener_knn(int(usuario_objetivo), user_ratings_global, k)
        for uid, sim in nuevos_vecinos:
            print(f"\nVecino: {uid} | Similitud: {sim:.4f}")


        # Tarea 4 (Cuadro de Complejidad Automatizado)
        ejecutar_experimento_completo(ruta_dataset, target_uid=usuario_objetivo)

    except FileNotFoundError:
        print(f"Por favor, asegúrate de que el archivo {ruta_dataset} existe en la carpeta actual.")