import time
import math

filepath = 'data/ratings.csv'

def cargar_datos_csv(filepath):
    """
    Carga el dataset de ratings.csv en un diccionario de diccionarios.
    Estructura: { usuario_id: { pelicula_id: rating } }
    """
    pref_usuarios = {}
    try:
        with open(filepath, 'r') as f:
            next(f)  # Saltar encabezado: userId,movieId,rating,timestamp
            for linea in f:
                partes = linea.strip().split(',')
                if len(partes) < 3:
                    continue
                u_id = partes[0]
                m_id = partes[1]
                rating = float(partes[2])
                
                if u_id not in pref_usuarios:
                    pref_usuarios[u_id] = {}
                pref_usuarios[u_id][m_id] = rating
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
    return pref_usuarios

def distancia_manhattan(usuario_u, usuario_v):
    """
    Calcula la distancia de Manhattan: suma de diferencias absolutas.
    Solo considera películas calificadas por ambos usuarios.
    """
    distancia = 0
    comunes = False
    
    # Iterar sobre las películas del usuario U
    for movie_id in usuario_u:
        if movie_id in usuario_v:
            distancia += math.fabs(usuario_u[movie_id] - usuario_v[movie_id])
            comunes = True
    # Si no hay películas en común, devolvemos un valor muy alto (infinito)
    return distancia if comunes else float('inf')

def distancia_euclidiana(usuario_u, usuario_v):
    """
    Calcula la distancia Euclidiana usando math.sqrt.
    """
    suma_cuadrados = 0
    comunes = False
    for movie_id in usuario_u:
        if movie_id in usuario_v:
            suma_cuadrados += (usuario_u[movie_id] - usuario_v[movie_id]) ** 2
            comunes = True  
    return math.sqrt(suma_cuadrados) if comunes else float('inf')

def similitud_coseno(usuario_u, usuario_v):
    """
    Calcula la similitud de coseno. 
    Retorna un valor de 'distancia' inversa para mantener consistencia con k-NN
    (donde menor valor significa más cercano).
    """
    dot_product = 0
    norm_u = 0
    norm_v = 0
    comunes = False
    
    # Solo calculamos sobre los ítems que ambos han calificado para ser más precisos
    # o sobre la unión para una norma formal. Aquí usaremos la norma de todos sus ratings.
    for rating in usuario_u.values():
        norm_u += rating ** 2
    for rating in usuario_v.values():
        norm_v += rating ** 2
        
    for movie_id, rating_u in usuario_u.items():
        if movie_id in usuario_v:
            dot_product += rating_u * usuario_v[movie_id]
            comunes = True
            
    if not comunes or norm_u == 0 or norm_v == 0:
        return float('inf')
    
    similitud = dot_product / (math.sqrt(norm_u) * math.sqrt(norm_v))
    
    # Como k-NN busca MINIMIZAR distancia, convertimos similitud (0 a 1) 
    # en una métrica de distancia (1 - similitud).
    return 1 - similitud

def obtener_k_vecinos(u_id, todos_usuarios, k=5, metrica='coseno'):
    """
    Tarea 1: Retorna los K vecinos más cercanos al usuario u_id según la métrica.
    """
    if u_id not in todos_usuarios: return []
    
    u_pref = todos_usuarios[u_id]
    
    # Mapeo de métricas
    metricas = {
        'manhattan': distancia_manhattan,
        'euclidiana': distancia_euclidiana,
        'coseno': similitud_coseno
    }
    func_dist = metricas.get(metrica, similitud_coseno)
    
    distancias = []
    for v_id, v_pref in todos_usuarios.items():
        if v_id == u_id: continue
        d = func_dist(u_pref, v_pref)
        if d != float('inf'):
            distancias.append((v_id, d))
            
    # Ordenar: menor distancia = más parecido
    distancias.sort(key=lambda x: x[1])
    return distancias[:k]

def recomendar_peliculas(u_id, todos_usuarios, k=5, umbral=3.0, metrica='coseno'):
    """
    Tarea 2: Recomienda películas no vistas basándose en los K vecinos.
    """
    vecinos = obtener_k_vecinos(u_id, todos_usuarios, k, metrica)
    if not vecinos: return []
    
    u_visto = todos_usuarios[u_id]
    recomendaciones = {} 
    
    for v_id, dist in vecinos:
        v_pref = todos_usuarios[v_id]
        for m_id, rating in v_pref.items():
            if m_id not in u_visto and rating > umbral:
                if m_id not in recomendaciones:
                    recomendaciones[m_id] = []
                recomendaciones[m_id].append(rating)
    
    resultado_final = []
    for m_id, ratings in recomendaciones.items():
        promedio = sum(ratings) / len(ratings)
        resultado_final.append((m_id, promedio))
        
    resultado_final.sort(key=lambda x: x[1], reverse=True)
    return resultado_final

# --- EJEMPLO DE EJECUCIÓN ---
if __name__ == "__main__":
    # 1. Cargar datos (Asegúrate de tener el archivo en la misma carpeta)
    print("Cargando dataset...")
    data = cargar_datos_csv('data/example.csv')
    
    if data:
        user_test = input("Ingrese el ID del usuario para probar (ej: '1'): ")
        k_val = 5
        
        print(f"\n--- COMPARATIVA DE MÉTRICAS PARA USUARIO {user_test} ---")
        
        for m in ['manhattan', 'euclidiana', 'coseno']:
            start = time.time()
            vecinos = obtener_k_vecinos(user_test, data, k=k_val, metrica=m)
            end = time.time()
            print(f"\nMétrica: {m.upper()} (Tiempo: {end-start:.4f}s)")
            for v in vecinos:
                print(f"Usuario ID: {v[0]} | Distancia: {v[1]:.4f}")
            print(f"Vecinos: {vecinos}")
            
        # Ejemplo de recomendación con la mejor métrica (Coseno)
        recoms = recomendar_peliculas(user_test, data, k=10, umbral=3.5, metrica='coseno')
        print(f"\nTop 5 Recomendaciones (Coseno) para Usuario {user_test}:")
        for m_id, score in recoms[:5]:
            print(f"Pelicula ID: {m_id} | Valor estimado: {score:.2f}")
    