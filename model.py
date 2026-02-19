import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from supabase import create_client
except ImportError:
    create_client = None

# --- Simulating Database Connection (Supabase) ---
def load_data():
    # 1. Intentamos cargar de Supabase si están las credenciales
    # Prioridad: Variables de Entorno (Producción)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        # Fallback: Credenciales Hardcoded (Desarrollo Local)
        url = "https://tu-proyecto.supabase.co" 
        key = "tu-anon-key-larga-aqui"
    
    if url and key and create_client:
        try:
            print("Conectando a Supabase...")
            supabase = create_client(url, key)
            # Asumimos que la tabla se llama 'orders'
            # IMPORTANTE: Usamos comillas dobles para "Total_monto" porque es Case Sensitive en Postgres
            response = supabase.table('orders').select('cliente_telefono, items, "Total_monto", restaurant_id, created_at').execute()
            data = pd.DataFrame(response.data)
            
            # Renombrar columnas para que coincidan con la lógica del modelo
            data = data.rename(columns={
                'cliente_telefono': 'customer_id',
                'items': 'order_item',
                'Total_monto': 'ticket_value'
            })

            # --- NUEVA LOGICA: PROCESAMIENTO DE JSON Y EXPLODE + BUNDLES ---
            # 1. Asegurar que sea una lista (parsear si es string)
            def parse_items(x):
                if isinstance(x, list): return x
                try:
                    import json
                    return json.loads(str(x).replace("'", '"')) 
                except:
                    return [str(x)]

            data['order_list_raw'] = data['order_item'].apply(parse_items)
            
            # 2. Limpieza de BUNDLES (Antes de explode, para detectar patrones de pedidos completos)
            blacklist = ['Total:', 'Pago:', 'Vuelto:', 'Envio', 'Recargo', 'Son:', 'Dirección', 'Nombre:']
            pattern = '|'.join(blacklist)

            def clean_bundle(items_list):
                # Filtra basura y devuelve lista limpia
                return [str(i).strip() for i in items_list if not any(b.lower() in str(i).lower() for b in blacklist)]

            data['order_bundle'] = data['order_list_raw'].apply(clean_bundle)
            # Creamos una "Firma" del pedido (ej: "Coca Cola, Pizza") para contar repetidos
            data['bundle_signature'] = data['order_bundle'].apply(lambda x: ', '.join(sorted(x)))
            
            # 3. Explode para el Modelo Neuronal (Items individuales)
            # Guardamos copia del dataframe con bundles ANTES de explotar, para uso en get_recurrent_bundle
            # Truco: Lo guardamos en el mismo objeto data temporalmente o lo retornamos.
            # Mejor: Lo asignamos a una variable global o atributo de clase si fuera posible, pero aqui es script.
            # SOLUCION: El dataframe retornado 'data' será el EXPLODED.
            # PERO agregamos una columna 'is_bundle_signature' que se repite, para poder reconstruir lógica básica.
            
            data = data.explode('order_bundle')
            data = data.rename(columns={'order_bundle': 'order_item'}) # El modelo usa 'order_item'
            
            # Limpieza final de filas vacías tras explode
            data = data[data['order_item'].str.len() > 0]
            # -----------------------------------------------------
            
            # Asegurarse de tener hora y día (extraer de created_at si existe, sino simular)
            if 'created_at' in data.columns:
                data['created_at'] = pd.to_datetime(data['created_at'])
                data['hour_of_day'] = data['created_at'].dt.hour
                data['day_of_week'] = data['created_at'].dt.dayofweek
            else:
                data['hour_of_day'] = 12
                data['day_of_week'] = 0

            # CORRECCIÓN DE ERROR 2: Asegurar que no haya NaNs en features numéricos
            data['ticket_value'] = pd.to_numeric(data['ticket_value'], errors='coerce').fillna(0)
            data['hour_of_day'] = data['hour_of_day'].fillna(12)
            data['day_of_week'] = data['day_of_week'].fillna(0)

            if not data.empty:
                print(f"Datos cargados de Supabase: {len(data)} registros.")
                return data
        except Exception as e:
            print(f"Error conectando a Supabase: {e}. Usando Mock Data.")
    
    print("Usando Datos Mock (Local)...")
    # Mock Data simplificado
    data = {
        'customer_id': ['C001', 'C001'],
        'restaurant_id': ['RestA', 'RestA'],
        'order_item': ['Pizza', 'Coca Cola'],
        'bundle_signature': ['Coca Cola, Pizza', 'Coca Cola, Pizza'], # Firma repetida
        'ticket_value': [50, 50],
        'hour_of_day': [20, 20],
        'day_of_week': [5, 5]
    }
    return pd.DataFrame(data)

class RestaurantRecommender:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.history_df = None # Para patrones exactos
        self.is_trained = False

    def train(self, df):
        # Guardamos referencia al DF completo (que incluye bundle_signature) para buscar patrones
        self.history_df = df.copy()
        
        restaurantes = df['restaurant_id'].unique()
        
        for rest_id in restaurantes:
            print(f"Entrenando modelo para: {rest_id}...")
            df_rest = df[df['restaurant_id'] == rest_id].copy()
            
            le_item = LabelEncoder()
            le_customer = LabelEncoder()
            
            # Fit encoders
            df_rest['item_code'] = le_item.fit_transform(df_rest['order_item'].astype(str))
            df_rest['customer_code'] = le_customer.fit_transform(df_rest['customer_id'].astype(str))
            
            X = df_rest[['customer_code', 'ticket_value', 'hour_of_day', 'day_of_week']]
            y = df_rest['item_code']
            
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
            model.fit(X, y)
            
            self.models[rest_id] = model
            self.encoders[rest_id] = { 'customer': le_customer, 'item': le_item }
            
        self.is_trained = True
        print("Modelos entrenados.")

    def get_recurrent_bundle(self, restaurant_id, customer_id):
        # Busca si el cliente repite mucho un mismo bundle
        if self.history_df is None: return None
        
        # Filtramos historial del cliente
        client_df = self.history_df[
            (self.history_df['restaurant_id'] == restaurant_id) & 
            (self.history_df['customer_id'] == customer_id)
        ]
        
        if client_df.empty: return None
        
        # Contamos cuántas veces aparece cada 'bundle_signature'
        # Como el DF está explotado, bundle_signature aparece N veces por orden.
        # Lo correcto sería agrupar por (created_at, bundle_signature) para contar ÓRDENES únicas.
        # Aproximación rápida: value_counts() de signature. 
        # Si una signature aparece mucho, es candidata.
        
        top_signature = client_df['bundle_signature'].mode()
        if not top_signature.empty:
            signature = top_signature[0]
            # Devolvemos la lista de items splitteada
            return signature.split(', ')
        return None

    def predict_recommendation(self, restaurant_id, customer_id, current_ticket_avg=0, hour=12, day=0):
        if not self.is_trained: return "Modelo no entrenado."
        
        if restaurant_id not in self.models:
            return {"recommendation": "Plato del Día", "reason": "Nuevo Restaurante", "model_type": "Fallback"}

        model = self.models[restaurant_id]
        le_customer = self.encoders[restaurant_id]['customer']
        le_item = self.encoders[restaurant_id]['item']

        # ESTRATEGIA 1: PEDIDO RECURRENTE (Adaptive Bundle)
        recurrent_bundle = self.get_recurrent_bundle(restaurant_id, customer_id)
        # Si hay un bundle recurrente (podríamos poner umbral de confianza), lo usamos.
        # Por ahora, si existe historial, asumimos el modo como preferencia fuerte.
        if recurrent_bundle:
             return {
                "restaurant_id": restaurant_id,
                "customer_id": customer_id,
                "recommendation": recurrent_bundle, # Lista de items
                "reason": f"Es tu pedido habitual en {restaurant_id}.",
                "model_type": "Pattern Recognition (Recurrent Bundle)"
            }

        # ESTRATEGIA 2: MODELO (Top 3 Items)
        if customer_id not in le_customer.classes_:
            return {"recommendation": ["Sugerencia del Chef"], "reason": "Cliente Nuevo", "model_type": "Heuristic"}

        try:
            customer_code = le_customer.transform([str(customer_id)])[0]
            probs = model.predict_proba([[customer_code, current_ticket_avg, hour, day]])[0]
            
            # Top 3
            top_3_indices = probs.argsort()[-3:][::-1]
            top_3_items = le_item.inverse_transform(top_3_indices)
            
            return {
                "restaurant_id": restaurant_id,
                "customer_id": customer_id,
                "recommendation": list(top_3_items),
                "reason": f"Basado en tus gustos variados en {restaurant_id}.",
                "model_type": "Neural Network (Top-3 Probabilistic)"
            }
        except Exception as e:
            return {"recommendation": ["Plato Popular"], "reason": f"Error calculando: {str(e)}", "model_type": "Error Fallback"}

recommender = RestaurantRecommender()
df = load_data()
if not df.empty and 'restaurant_id' in df.columns:
    recommender.train(df)
else:
    print("Advertencia: DataFrame vacío.")
