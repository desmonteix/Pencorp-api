# ============================================================
# MODELO PREDICTIVO HÍBRIDO - Pencorp
# ============================================================
# Este archivo contiene el motor de recomendación de productos
# para restaurantes. Usa DOS estrategias:
#
#   1. COLLABORATIVE FILTERING (CF) — Siempre activo
#      Compara el historial de compras entre clientes.
#      "Clientes que pidieron lo mismo que tú, también pidieron X"
#
#   2. NEURAL COLLABORATIVE FILTERING (NCF) — Se activa con 1000+ pedidos
#      Red neuronal con embeddings que aprende patrones profundos
#      de preferencia por usuario y producto.
#
# El sistema elige automáticamente cuál usar según la cantidad
# de datos disponibles por restaurante.
# ============================================================

import os
import json
import pandas as pd          # Manejo de tablas de datos (DataFrames)
import numpy as np            # Operaciones matemáticas con matrices
from sklearn.metrics.pairwise import cosine_similarity  # Mide similitud entre usuarios
from collections import defaultdict  # Diccionario con valor por defecto

# --- Conexión a base de datos (Supabase) ---
try:
    from supabase import create_client  # Cliente para leer pedidos de la BD
except ImportError:
    create_client = None

# --- PyTorch para la red neuronal (NCF) ---
# Si PyTorch no está instalado, el sistema funciona igual pero solo con CF
try:
    import torch
    import torch.nn as nn        # Para definir capas de la red neuronal
    import torch.optim as optim  # Optimizador (Adam) para entrenar la red
    from torch.utils.data import DataLoader, TensorDataset  # Para procesar datos en lotes
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch no disponible. NCF deshabilitado (solo Collaborative Filtering).")

# ============================================================
# SECCIÓN 1: CARGA DE DATOS DESDE SUPABASE
# ============================================================
# Esta función se conecta a Supabase, descarga todos los pedidos
# de la tabla 'orders', y los transforma en un formato limpio
# donde cada fila = 1 producto individual de 1 pedido.
#
# Ejemplo: Si un pedido tenía ["Pizza", "Coca Cola"], se crean
# 2 filas separadas, una por cada producto.
# ============================================================
last_load_error = None  # Almacena el último error para debugging

def load_data():
    global last_load_error
    last_load_error = None

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        last_load_error = "Credenciales de Supabase no configuradas (ENV VARS faltantes)."
        print(last_load_error)
        return pd.DataFrame()

    if url and key and create_client:
        try:
            print(f"Conectando a Supabase: {url}...")
            supabase = create_client(url, key)
            response = supabase.table('orders').select(
                'cliente_telefono, items, "Total_monto", restaurant_id, created_at'
            ).execute()
            data = pd.DataFrame(response.data)

            data = data.rename(columns={
                'cliente_telefono': 'customer_id',
                'items': 'order_item',
                'Total_monto': 'ticket_value'
            })

            # --- Descomponer el JSON de items ---
            # El campo 'items' puede venir en varios formatos:
            #   - Lista directa: ["Pizza", "Coca Cola"]
            #   - Diccionario: {"items": ["Pizza", "Coca Cola"]}
            #   - String de texto: "Pizza, Coca Cola"
            # Esta función normaliza todos los formatos a una lista Python.
            def parse_items(x):
                if isinstance(x, list): return x       # Ya es lista, perfecto
                if pd.isna(x): return []                # Valor vacío/nulo
                try:
                    s = str(x).replace("'", '"')        # Reemplazar comillas simples
                    parsed = json.loads(s)               # Intentar parsear como JSON
                    if isinstance(parsed, dict) and 'items' in parsed:
                        return parsed['items']           # Caso: {"items": [...]}
                    if isinstance(parsed, list):
                        return parsed                    # Caso: [...]
                    return [str(parsed)]                 # Caso: valor único
                except:
                    return [str(x)]                      # Fallback: tratar como texto

            data['order_list_raw'] = data['order_item'].apply(parse_items)

            # --- Normalize customer IDs ---
            def normalize_phone(phone):
                if pd.isna(phone): return "UNKNOWN"
                return ''.join(filter(str.isdigit, str(phone)))

            data['customer_id_clean'] = data['customer_id'].apply(normalize_phone)

            # --- Limpiar items (quitar metadatos/basura del JSON) ---
            # El JSON del pedido a veces incluye líneas como "Total: S/55"
            # o "Pago: Yape" que NO son productos. Las filtramos aquí.
            blacklist = ['Total:', 'Pago:', 'Vuelto:', 'Envio', 'Recargo', 'Son:',
                         'Dirección', 'Nombre:', 'Fecha:', 'Mesa:']

            def clean_bundle(items_list):
                """Limpia la lista de items removiendo basura y normalizando formato."""
                if not isinstance(items_list, list): return []
                cleaned = []
                for i in items_list:
                    s = str(i).strip()
                    if s.startswith('*'): s = s[1:].strip()  # Quitar asteriscos: "*Pizza" → "Pizza"
                    parts = s.split(' ', 1)
                    # Quitar cantidades: "2x Pizza" → "Pizza"
                    if len(parts) > 1 and parts[0].endswith('x') and parts[0][:-1].isdigit():
                        s = parts[1]
                    # Solo agregar si NO es metadata (Total, Pago, etc.)
                    if not any(b.lower() in s.lower() for b in blacklist):
                        cleaned.append(s)
                return cleaned

            data['order_bundle'] = data['order_list_raw'].apply(clean_bundle)
            data['bundle_signature'] = data['order_bundle'].apply(lambda x: ', '.join(sorted(x)))

            # --- Explode: one row per item ---
            if 'order_item' in data.columns:
                data = data.drop(columns=['order_item'])
            data = data.explode('order_bundle')
            data = data.rename(columns={'order_bundle': 'order_item'})
            data = data.dropna(subset=['order_item'])
            data = data[data['order_item'].str.len() > 0]

            # --- Time features ---
            if 'created_at' in data.columns:
                data['created_at'] = pd.to_datetime(data['created_at'], utc=True)
                data['hour_of_day'] = data['created_at'].dt.hour
                data['day_of_week'] = data['created_at'].dt.dayofweek
            else:
                data['hour_of_day'] = 12
                data['day_of_week'] = 0

            data['ticket_value'] = pd.to_numeric(data['ticket_value'], errors='coerce').fillna(0)

            if data.empty:
                last_load_error = "Conexión exitosa, pero tabla 'orders' vacía."
                print(last_load_error)
            else:
                print(f"✅ Datos cargados: {len(data)} registros.")

            return data

        except Exception as e:
            last_load_error = f"Error conectando a Supabase: {str(e)}"
            print(last_load_error)
            return pd.DataFrame()

    last_load_error = "Error desconocido (library missing?)"
    return pd.DataFrame()


# ============================================================
# SECCIÓN 2: COLLABORATIVE FILTERING (CF)
# ============================================================
# ¿Cómo funciona?
# 1. Construye una MATRIZ USUARIO x PRODUCTO con frecuencias.
#    Ejemplo:
#                Pizza  Hamburguesa  Coca Cola
#    Carlos        5        3          2
#    María         4        0          5
#    José          5        4          1
#
# 2. Calcula SIMILITUD COSENO entre usuarios.
#    Carlos y José son parecidos porque ambos piden mucha Pizza.
#
# 3. Para recomendar a Carlos: mira qué piden los usuarios
#    más parecidos a él, y sugiere esos productos.
#
# 4. Mezcla 60% recomendaciones colaborativas + 40% historial
#    personal del usuario.
# ============================================================
class CollaborativeFilter:
    def __init__(self):
        self.user_item_matrix = None  # Matriz de frecuencias (numpy array)
        self.user_sim = None          # Matriz de similitud entre usuarios
        self.user_ids = []            # Lista ordenada de IDs de usuario
        self.item_names = []          # Lista ordenada de nombres de productos
        self.user_to_idx = {}         # Mapeo: ID usuario → índice en la matriz
        self.item_to_idx = {}         # Mapeo: nombre producto → índice en la matriz
        self.top_sellers = []         # Los 5 productos más vendidos (fallback)
        self.user_profiles = {}       # Estadísticas por usuario (ticket, frecuencia, etc.)
        self.n_users = 0              # Total de usuarios únicos
        self.n_items = 0              # Total de productos únicos

    def fit(self, df):
        """
        Entrena el Collaborative Filter con datos históricos.
        Construye la matriz usuario-producto y calcula similitudes.
        """
        # Paso 1: Contar cuántas veces cada usuario pidió cada producto
        interactions = df.groupby(['customer_id_clean', 'order_item']).size().reset_index(name='freq')

        # Paso 2: Crear mapeos (ID → posición en la matriz)
        self.user_ids = sorted(interactions['customer_id_clean'].unique().tolist())
        self.item_names = sorted(interactions['order_item'].unique().tolist())
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_to_idx = {item: i for i, item in enumerate(self.item_names)}
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_names)

        # Paso 3: Llenar la matriz con las frecuencias
        matrix = np.zeros((self.n_users, self.n_items))
        for _, row in interactions.iterrows():
            u = self.user_to_idx[row['customer_id_clean']]
            i = self.item_to_idx[row['order_item']]
            matrix[u, i] = row['freq']

        self.user_item_matrix = matrix

        # Paso 4: Calcular similitud coseno entre todos los pares de usuarios
        # Resultado: matriz NxN donde [i][j] = qué tan parecidos son usuario i y j
        # Valores van de 0 (nada parecidos) a 1 (idénticos)
        if self.n_users > 1:
            self.user_sim = cosine_similarity(matrix)
            np.fill_diagonal(self.user_sim, 0)  # Un usuario no es "similar a sí mismo"
        else:
            self.user_sim = np.zeros((1, 1))

        # Paso 5: Calcular los productos más vendidos (para clientes nuevos)
        self.top_sellers = df['order_item'].value_counts().head(5).index.tolist()

        # Paso 6: Construir perfil estadístico de cada usuario
        # Esto nos da info como: cuántos pedidos ha hecho, su ticket promedio, etc.
        for uid in self.user_ids:
            user_df = df[df['customer_id_clean'] == uid]
            if 'created_at' in user_df.columns and user_df['created_at'].notna().any():
                orders = user_df.drop_duplicates(subset=['created_at'])
                n_orders = len(orders)
                avg_ticket = orders['ticket_value'].mean()
                last_order = user_df['created_at'].max()
                now = pd.Timestamp.now(tz='UTC')
                days_since = (now - last_order).days if pd.notna(last_order) else 999
            else:
                n_orders = 1
                avg_ticket = user_df['ticket_value'].mean()
                days_since = 0

            self.user_profiles[uid] = {
                'total_orders': int(n_orders),
                'avg_ticket': round(float(avg_ticket), 2),
                'unique_items': int(user_df['order_item'].nunique()),
                'days_since_last': int(days_since),
                'favorite_items': user_df['order_item'].value_counts().head(3).index.tolist()
            }

        print(f"  CF: {self.n_users} usuarios, {self.n_items} items")

    def recommend(self, user_id_clean, n=3):
        """
        Genera recomendaciones personalizadas para un usuario.
        Retorna: lista de (producto, score) y una razón explicativa.
        """
        # CASO 1: Cliente nuevo (no existe en nuestro historial)
        # → Le recomendamos los productos más vendidos del restaurante
        if user_id_clean not in self.user_to_idx:
            items = [(it, 1.0) for it in self.top_sellers[:n]]
            return items, "Sugerencias más populares (Cliente Nuevo)"

        user_idx = self.user_to_idx[user_id_clean]
        user_history = self.user_item_matrix[user_idx]

        # CASO 2: Solo hay 1 usuario en el sistema (no se puede comparar)
        # → Le recomendamos sus propios favoritos
        if self.n_users <= 1:
            top_idx = user_history.argsort()[-n:][::-1]
            items = [(self.item_names[i], float(user_history[i])) for i in top_idx if user_history[i] > 0]
            if not items:
                items = [(it, 1.0) for it in self.top_sellers[:n]]
            return items, "Tus favoritos"

        # CASO 3: Múltiples usuarios — Filtrado Colaborativo real
        # Buscamos los K usuarios más similares y ponderamos sus preferencias
        sims = self.user_sim[user_idx]
        k = min(10, self.n_users - 1)  # Máximo 10 vecinos
        top_neighbors = sims.argsort()[-k:][::-1]  # Índices de los más similares
        top_sims = sims[top_neighbors]              # Sus scores de similitud

        # Score colaborativo: suma ponderada de lo que piden los vecinos
        # Si un vecino muy similar pide mucha Pizza, Pizza tendrá score alto
        collab_scores = np.zeros(self.n_items)
        for neighbor, sim in zip(top_neighbors, top_sims):
            if sim > 0:
                collab_scores += sim * self.user_item_matrix[neighbor]

        # Score personal: qué ha pedido este usuario antes (normalizado 0-1)
        personal_scores = np.zeros(self.n_items)
        if user_history.max() > 0:
            personal_scores = user_history / user_history.max()

        # MEZCLA FINAL: 60% colaborativo + 40% personal
        # Esto balancea "lo que le gusta a gente similar" con "lo que ya pide"
        max_collab = collab_scores.max() if collab_scores.max() > 0 else 1
        blended = 0.6 * (collab_scores / max_collab) + 0.4 * personal_scores

        # Seleccionar los N productos con mayor score
        top_idx = blended.argsort()[-n:][::-1]
        items = [(self.item_names[i], round(float(blended[i]), 3)) for i in top_idx if blended[i] > 0]

        if not items:
            items = [(it, 1.0) for it in self.top_sellers[:n]]
            return items, "Top Sellers (sin datos suficientes)"

        profile = self.user_profiles.get(user_id_clean, {})
        n_ord = profile.get('total_orders', 0)
        reason = f"Basado en tus {n_ord} pedidos anteriores y gustos de clientes similares"
        return items, reason


# ============================================================
# SECCIÓN 3: NEURAL COLLABORATIVE FILTERING (NCF)
# ============================================================
# Se activa SOLO cuando un restaurante tiene 1000+ pedidos.
#
# ¿Cómo funciona?
# - Cada usuario se representa como un VECTOR de 16 números
#   ("embedding") que la red neuronal APRENDE automáticamente.
# - Cada producto también tiene su propio embedding de 16 números.
# - La red concatena ambos embeddings + contexto (hora, día, ticket)
#   y pasa todo por capas densas para predecir:
#   "¿Qué tan probable es que este usuario pida este producto?"
#
# Arquitectura:
#   [User Embed (16)] + [Item Embed (16)] + [hora, día, ticket]
#                          ↓
#                    Dense(64) + ReLU
#                    Dense(32) + ReLU
#                    Dense(1) + Sigmoid → Probabilidad 0-1
# ============================================================
if TORCH_AVAILABLE:
    class NCFModel(nn.Module):
        """Red neuronal para predecir la probabilidad de que un usuario pida un producto."""
        def __init__(self, n_users, n_items, embed_dim=16, context_dim=3):
            super().__init__()
            # Capa de embedding: convierte ID de usuario en un vector de 16 dimensiones
            # La red APRENDE estos vectores durante el entrenamiento
            self.user_embed = nn.Embedding(n_users, embed_dim)
            # Lo mismo para productos
            self.item_embed = nn.Embedding(n_items, embed_dim)
            # Red neuronal (MLP): procesa los embeddings concatenados
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2 + context_dim, 64),  # Entrada: 16+16+3 = 35
                nn.ReLU(),            # Activación no-lineal
                nn.Dropout(0.2),      # Previene sobreajuste (apaga 20% de neuronas al azar)
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),     # Salida: 1 número
                nn.Sigmoid()          # Lo convierte en probabilidad 0-1
            )
            # Inicialización Xavier: pone pesos iniciales inteligentes
            nn.init.xavier_uniform_(self.user_embed.weight)
            nn.init.xavier_uniform_(self.item_embed.weight)

        def forward(self, user_ids, item_ids, context):
            """Pasa datos por la red. Retorna probabilidad de interacción."""
            u = self.user_embed(user_ids)   # Buscar embedding del usuario
            i = self.item_embed(item_ids)   # Buscar embedding del producto
            x = torch.cat([u, i, context], dim=1)  # Concatenar todo
            return self.mlp(x).squeeze()    # Pasar por la red neuronal


class NCFEngine:
    """
    Motor de entrenamiento y predicción para la red neuronal NCF.
    Maneja la creación de datos de entrenamiento, el entrenamiento
    de la red, y la generación de recomendaciones.
    """
    def __init__(self):
        self.model = None          # La red neuronal (NCFModel)
        self.user_to_idx = {}      # Mapeo: ID usuario → índice
        self.item_to_idx = {}      # Mapeo: nombre producto → índice
        self.idx_to_item = {}      # Mapeo inverso: índice → nombre producto
        self.n_items = 0           # Total de productos
        self.top_sellers = []      # Fallback para clientes nuevos
        self.max_ticket = 1        # Ticket máximo (para normalizar)
        self.is_trained = False    # ¿Ya se entrenó?

    def fit(self, df, epochs=20, lr=0.001, neg_ratio=4):
        """
        Entrena la red neuronal NCF.
        - epochs: cuántas veces recorre todos los datos (20 por defecto)
        - lr: learning rate (velocidad de aprendizaje)
        - neg_ratio: por cada pedido real, crear N ejemplos negativos
          (productos que el usuario NO pidió, para que aprenda a diferenciar)
        """
        if not TORCH_AVAILABLE:
            return

        # Crear mapeos de IDs a índices numéricos
        users = sorted(df['customer_id_clean'].unique().tolist())
        items = sorted(df['order_item'].unique().tolist())
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}
        self.n_items = len(items)
        self.top_sellers = df['order_item'].value_counts().head(5).index.tolist()

        # --- Construir muestras POSITIVAS ---
        # Cada fila del DataFrame es una interacción real (usuario SÍ pidió ese producto)
        # Normalizamos hora (0-1), día (0-1) y ticket para que la red trabaje mejor
        positives = []
        for _, row in df.iterrows():
            uid = self.user_to_idx.get(row['customer_id_clean'])
            iid = self.item_to_idx.get(row['order_item'])
            if uid is not None and iid is not None:
                h = float(row.get('hour_of_day', 12)) / 23.0   # Normalizar hora a 0-1
                d = float(row.get('day_of_week', 0)) / 6.0     # Normalizar día a 0-1
                t = float(row.get('ticket_value', 0))           # Ticket (se normaliza abajo)
                positives.append((uid, iid, h, d, t, 1.0))     # 1.0 = SÍ pidió este producto

        if not positives:
            return

        # Normalizar tickets dividiéndolos por el máximo
        tickets = [p[4] for p in positives]
        self.max_ticket = max(tickets) if max(tickets) > 0 else 1
        positives = [(u, i, h, d, t / self.max_ticket, lb) for u, i, h, d, t, lb in positives]

        # --- Construir muestras NEGATIVAS ---
        # Para que la red aprenda, necesita ver productos que el usuario NO pidió
        # Por cada muestra positiva, creamos 'neg_ratio' muestras negativas aleatorias
        # con label 0.0 ("este usuario NO pidió este producto")
        all_items = set(range(self.n_items))
        user_items = defaultdict(set)
        for u, i, *_ in positives:
            user_items[u].add(i)  # Registrar qué productos SÍ pidió cada usuario

        negatives = []
        for u, i, h, d, t, _ in positives:
            neg_pool = list(all_items - user_items[u])  # Productos que NO pidió
            if neg_pool:
                for _ in range(min(neg_ratio, len(neg_pool))):
                    neg_i = int(np.random.choice(neg_pool))
                    negatives.append((u, neg_i, h, d, t, 0.0))  # 0.0 = NO pidió

        # Combinar positivos + negativos y mezclar aleatoriamente
        all_samples = positives + negatives
        np.random.shuffle(all_samples)

        users_t = torch.LongTensor([s[0] for s in all_samples])
        items_t = torch.LongTensor([s[1] for s in all_samples])
        ctx_t = torch.FloatTensor([[s[2], s[3], s[4]] for s in all_samples])
        labels_t = torch.FloatTensor([s[5] for s in all_samples])

        dataset = TensorDataset(users_t, items_t, ctx_t, labels_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        n_users = len(users)
        self.model = NCFModel(n_users, self.n_items, embed_dim=16, context_dim=3)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for bu, bi, bc, bl in loader:
                optimizer.zero_grad()
                preds = self.model(bu, bi, bc)
                loss = criterion(preds, bl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"    NCF Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        self.model.eval()
        self.is_trained = True
        print(f"  NCF: Entrenado ({len(positives)} pos + {len(negatives)} neg samples)")

    def recommend(self, user_id_clean, hour=12, day=0, ticket_avg=0, n=3):
        """
        Genera recomendaciones usando la red neuronal.
        Puntea TODOS los productos del menú para este usuario
        y devuelve los N con mayor probabilidad.
        """
        if not self.is_trained or not TORCH_AVAILABLE:
            return None, None

        # Cliente nuevo: la red no lo conoce, devolver Top Sellers
        if user_id_clean not in self.user_to_idx:
            items = [(it, 1.0) for it in self.top_sellers[:n]]
            return items, "Top Sellers (Cliente Nuevo para NCF)"

        uid = self.user_to_idx[user_id_clean]

        # Evaluar todos los productos del menú para este usuario
        with torch.no_grad():  # Desactivar gradientes (no estamos entrenando)
            u_t = torch.LongTensor([uid] * self.n_items)     # Repetir usuario N veces
            i_t = torch.LongTensor(list(range(self.n_items))) # Todos los productos
            ctx = torch.FloatTensor([                          # Contexto actual
                [hour / 23.0, day / 6.0, min(ticket_avg / self.max_ticket, 1.0)]
            ] * self.n_items)
            scores = self.model(u_t, i_t, ctx).numpy()  # Probabilidades 0-1

        # Seleccionar los N productos con mayor probabilidad
        top_idx = scores.argsort()[-n:][::-1]
        items = [(self.idx_to_item[i], round(float(scores[i]), 3)) for i in top_idx]
        conf = scores[top_idx[0]]
        reason = f"Predicción de Red Neuronal personalizada (confianza: {conf:.0%})"
        return items, reason


# ============================================================
# SECCIÓN 4: ORQUESTADOR HÍBRIDO
# ============================================================
# Esta clase decide QUÉ motor usar para cada restaurante:
#   - Si tiene < 1000 pedidos → Collaborative Filtering (CF)
#   - Si tiene >= 1000 pedidos → Neural Collaborative Filtering (NCF)
#
# Entrena un modelo SEPARADO por cada restaurante, porque cada
# restaurante tiene su propio menú y sus propios clientes.
# ============================================================
class HybridRecommender:
    NCF_THRESHOLD = 1000  # Mínimo de registros para activar la red neuronal

    def __init__(self):
        self.cf_models = {}        # Un CollaborativeFilter por restaurante
        self.ncf_models = {}       # Un NCFEngine por restaurante (solo si hay 1000+ datos)
        self.restaurant_stats = {} # Estadísticas de cada restaurante
        self.is_trained = False    # ¿Algún modelo está entrenado?

    def train(self, df):
        """Entrena modelos para TODOS los restaurantes encontrados en los datos."""
        if df.empty:
            print("DataFrame vacío. No se puede entrenar.")
            self.is_trained = False
            return

        # Iterar por cada restaurante y entrenar su modelo independiente
        for rest_id in df['restaurant_id'].unique():
            print(f"\n--- Entrenando para: {rest_id} ---")
            df_rest = df[df['restaurant_id'] == rest_id].copy()
            n_records = len(df_rest)

            # Guardar estadísticas del restaurante
            self.restaurant_stats[rest_id] = {
                'total_records': n_records,
                'unique_users': int(df_rest['customer_id_clean'].nunique()),
                'unique_items': int(df_rest['order_item'].nunique()),
                'engine': 'CF'  # Por defecto usa CF
            }

            # SIEMPRE entrenar Collaborative Filtering (funciona con pocos datos)
            cf = CollaborativeFilter()
            cf.fit(df_rest)
            self.cf_models[rest_id] = cf

            # Entrenar NCF SOLO si hay suficientes datos Y PyTorch está disponible
            if n_records >= self.NCF_THRESHOLD and TORCH_AVAILABLE:
                print(f"  {n_records} registros >= {self.NCF_THRESHOLD}: Activando NCF...")
                ncf = NCFEngine()
                ncf.fit(df_rest)
                if ncf.is_trained:
                    self.ncf_models[rest_id] = ncf
                    self.restaurant_stats[rest_id]['engine'] = 'NCF + CF'
            else:
                why = "PyTorch no disponible" if not TORCH_AVAILABLE else f"{n_records} < {self.NCF_THRESHOLD}"
                print(f"  NCF desactivado ({why}). Solo CF.")

        self.is_trained = True
        print("\n✅ Modelos entrenados exitosamente.")

    def predict_recommendation(self, restaurant_id, customer_id, current_ticket_avg=0, hour=12, day=0):
        """
        FUNCIÓN PRINCIPAL: genera recomendaciones para un cliente.
        Esta es la función que llama el endpoint /predict de la API.

        Parámetros:
        - restaurant_id: ID del restaurante
        - customer_id: teléfono/ID del cliente
        - current_ticket_avg: ticket promedio del cliente
        - hour: hora actual (0-23)
        - day: día de la semana (0=Lunes, 6=Domingo)

        Retorna un dict con: recommendation, reason, model_type, model_details
        """
        global last_load_error
        # Normalizar ID del cliente (solo dígitos, quitar +, espacios, etc.)
        customer_clean = ''.join(filter(str.isdigit, str(customer_id)))

        # Respuesta por defecto (se actualiza según el caso)
        response = {
            "restaurant_id": restaurant_id,
            "customer_id": customer_id,
            "recommendation": ["Plato del Día"],
            "reason": "Inicio",
            "model_type": "Unknown",
            "model_details": {}
        }

        # --- Auto-reentrenamiento si el modelo no está listo ---
        # Esto pasa la primera vez que alguien hace un request
        # si al iniciar no había datos en Supabase
        if not self.is_trained:
            print("Modelo no entrenado. Cargando datos...")
            try:
                new_df = load_data()
                if not new_df.empty:
                    self.train(new_df)
            except Exception as e:
                print(f"Error al reentrenar: {e}")

        if not self.is_trained:
            err = last_load_error or "Modelo sin datos (DB Vacía)"
            response.update({"reason": f"No entrenado: {err}", "model_type": "Error"})
            return response

        # --- Verificar que el restaurante exista ---
        if restaurant_id not in self.cf_models:
            try:
                new_df = load_data()
                if not new_df.empty:
                    self.train(new_df)
            except Exception:
                pass
            if restaurant_id not in self.cf_models:
                response.update({"reason": "Restaurante no encontrado", "model_type": "Fallback"})
                return response

        stats = self.restaurant_stats.get(restaurant_id, {})

        # --- ESTRATEGIA 1: Intentar NCF (red neuronal) si está disponible ---
        if restaurant_id in self.ncf_models:
            ncf = self.ncf_models[restaurant_id]
            items, reason = ncf.recommend(customer_clean, hour, day, current_ticket_avg, n=3)
            if items:
                response.update({
                    "recommendation": [it[0] for it in items],
                    "reason": reason,
                    "model_type": "Neural Collaborative Filtering (NCF)",
                    "model_details": {
                        "engine": "NCF",
                        "total_records": stats.get('total_records', 0),
                        "scores": {it[0]: it[1] for it in items}
                    }
                })
                return response

        # --- ESTRATEGIA 2: Collaborative Filtering (siempre disponible) ---
        cf = self.cf_models[restaurant_id]
        items, reason = cf.recommend(customer_clean, n=3)
        profile = cf.user_profiles.get(customer_clean, {})  # Perfil del usuario

        response.update({
            "recommendation": [it[0] for it in items],
            "reason": reason,
            "model_type": "Collaborative Filtering",
            "model_details": {
                "engine": "CF",
                "total_records": stats.get('total_records', 0),
                "user_profile": profile if profile else "Cliente Nuevo",
                "scores": {it[0]: it[1] for it in items}
            }
        })
        return response

    def get_debug_info(self):
        """Devuelve información de diagnóstico sobre el estado del modelo."""
        return {
            "is_trained": self.is_trained,
            "torch_available": TORCH_AVAILABLE,
            "ncf_threshold": self.NCF_THRESHOLD,
            "restaurants": {
                rid: {
                    **stats,
                    "has_ncf": rid in self.ncf_models,
                    "cf_users": self.cf_models[rid].n_users if rid in self.cf_models else 0,
                    "cf_items": self.cf_models[rid].n_items if rid in self.cf_models else 0,
                }
                for rid, stats in self.restaurant_stats.items()
            }
        }


# ============================================================
# INICIALIZACIÓN AL ARRANCAR EL SERVIDOR
# ============================================================
# Cuando FastAPI arranca, este código se ejecuta automáticamente:
# 1. Crea el recomendador híbrido
# 2. Intenta cargar datos de Supabase
# 3. Si hay datos, entrena los modelos
# 4. Si no hay datos, el modelo se entrenará en el primer request (lazy loading)
# ============================================================
recommender = HybridRecommender()
df = load_data()
if not df.empty and 'restaurant_id' in df.columns:
    recommender.train(df)
else:
    print("⚠️ Advertencia: DataFrame vacío al iniciar.")
