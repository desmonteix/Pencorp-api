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
                if pd.isna(x): return []
                try:
                    s = str(x).replace("'", '"')
                    import json
                    parsed = json.loads(s)
                    # CASO: {"items": ["Pizza", ...]} -> Extraer lista interna
                    if isinstance(parsed, dict) and 'items' in parsed:
                        return parsed['items']
                    # CASO: Lista directa -> Devolver
                    if isinstance(parsed, list):
                        return parsed
                    return [str(parsed)]
                except:
                    return [str(x)]

            data['order_list_raw'] = data['order_item'].apply(parse_items)
            
            # 2. Limpieza de BUNDLES 
            # Normalizamos ID de cliente: Quitamos espacios, +, y guiones para que coincida siempre
            def normalize_phone(phone):
                if pd.isna(phone): return "UNKNOWN"
                return ''.join(filter(str.isdigit, str(phone)))

            data['customer_id_clean'] = data['customer_id'].apply(normalize_phone)
            
            blacklist = ['Total:', 'Pago:', 'Vuelto:', 'Envio', 'Recargo', 'Son:', 'Dirección', 'Nombre:', 'Fecha:', 'Mesa:']
            pattern = '|'.join(blacklist)

            def clean_bundle(items_list):
                if not isinstance(items_list, list): return []
                # Filtra basura y devuelve lista limpia
                cleaned = []
                for i in items_list:
                    s = str(i).strip()
                    # Quitar asteriscos iniciales (ej: "*1x Pizza")
                    if s.startswith('*'): s = s[1:].strip()
                    # Quitar "1x ", "2x " del inicio si existe
                    parts = s.split(' ', 1)
                    if len(parts) > 1 and parts[0].endswith('x') and parts[0][:-1].isdigit():
                        s = parts[1]
                    
                    if not any(b.lower() in s.lower() for b in blacklist):
                        cleaned.append(s)
                return cleaned

            data['order_bundle'] = data['order_list_raw'].apply(clean_bundle)
            data['bundle_signature'] = data['order_bundle'].apply(lambda x: ', '.join(sorted(x)))
            
            # 3. Explode
            data = data.explode('order_bundle')
            data = data.rename(columns={'order_bundle': 'order_item'}) 
            data = data[data['order_item'].str.len() > 0]
            
            # Asegurarse de tener hora y día
            if 'created_at' in data.columns:
                data['created_at'] = pd.to_datetime(data['created_at'])
                data['hour_of_day'] = data['created_at'].dt.hour
                data['day_of_week'] = data['created_at'].dt.dayofweek
            else:
                data['hour_of_day'] = 12
                data['day_of_week'] = 0

            data['ticket_value'] = pd.to_numeric(data['ticket_value'], errors='coerce').fillna(0)
            data['hour_of_day'] = data['hour_of_day'].fillna(12)
            data['day_of_week'] = data['day_of_week'].fillna(0)

            # --- CORRECCIÓN IMPORTANTE ---
            # Si conecta a Supabase pero no hay datos (Base de datos vacía), devolvemos DataFrame vacío.
            # NO usamos Mock Data para no "inventar" recomendaciones al usuario.
            if data.empty:
                print("Conexión exitosa a Supabase, pero la tabla 'orders' está vacía.")
            else:
                print(f"Datos cargados de Supabase: {len(data)} registros.")
            
            return data

            if data.empty:
                print("Conexión exitosa a Supabase, pero la tabla 'orders' está vacía.")
            else:
                print(f"Datos cargados de Supabase: {len(data)} registros.")
            
            return data

        except Exception as e:
            print(f"Error conectando a Supabase: {e}")
            # --- CORRECCIÓN FINAL ---
            # Si falla la conexión, NO usamos Mock Data. 
            # Devolvemos un DataFrame vacío para que el sistema reporte "Error" o "Sin Datos".
            return pd.DataFrame()
    
    # Si no hay credenciales configured
    print("No se encontraron credenciales de Supabase (Variables de Entorno).")
    return pd.DataFrame()

class RestaurantRecommender:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.history_df = None 
        self.top_sellers = {} # { 'RestA': ['Pizza', 'Coca'] }
        self.is_trained = False

    def train(self, df):
        if df.empty:
            print("DataFrame vacío. No se puede entrenar el modelo.")
            self.is_trained = False
            return

        self.history_df = df.copy()
        
        restaurantes = df['restaurant_id'].unique()
        
        for rest_id in restaurantes:
            print(f"Entrenando modelo para: {rest_id}...")
            df_rest = df[df['restaurant_id'] == rest_id].copy()
            
            # Calcular Top Sellers (Globales para este restaurante)
            # Excluyendo items vacíos
            if 'order_item' in df_rest.columns:
                 top = df_rest['order_item'].value_counts().head(3).index.tolist()
                 self.top_sellers[rest_id] = top
            else:
                 self.top_sellers[rest_id] = ["Menú de la Casa"]

            le_item = LabelEncoder()
            le_customer = LabelEncoder() # Usaremos customer_id_clean
            
            # Fit encoders
            df_rest['item_code'] = le_item.fit_transform(df_rest['order_item'].astype(str))
            df_rest['customer_code'] = le_customer.fit_transform(df_rest['customer_id_clean'].astype(str))
            
            X = df_rest[['customer_code', 'ticket_value', 'hour_of_day', 'day_of_week']]
            y = df_rest['item_code']
            
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
            model.fit(X, y)
            
            self.models[rest_id] = model
            self.encoders[rest_id] = { 'customer': le_customer, 'item': le_item }
            
        self.is_trained = True
        print("Modelos entrenados.")

    def get_recurrent_bundle(self, restaurant_id, customer_clean):
        if self.history_df is None: return None
        
        client_df = self.history_df[
            (self.history_df['restaurant_id'] == restaurant_id) & 
            (self.history_df['customer_id_clean'] == customer_clean)
        ]
        
        if client_df.empty: return None
        
        # Filtramos bundles vacíos o nulos
        client_df = client_df[client_df['bundle_signature'].str.len() > 0]
        if client_df.empty: return None

        top_signature = client_df['bundle_signature'].mode()
        if not top_signature.empty:
            signature = top_signature[0]
            if pd.isna(signature): return None
            return signature.split(', ')
        return None

    def predict_recommendation(self, restaurant_id, customer_id, current_ticket_avg=0, hour=12, day=0):
        # Normalizar ID de entrada
        customer_clean = ''.join(filter(str.isdigit, str(customer_id)))
        
        response = {
            "restaurant_id": restaurant_id,
            "customer_id": customer_id, # Devolvemos el original para el usuario
            "recommendation": ["Plato del Día"],
            "reason": "Inicio",
            "model_type": "Unknown"
        }

        if not self.is_trained: 
            response.update({"reason": "Modelo no entrenado", "model_type": "Error"})
            return response
        
        # Fallback si no existe el restaurante
        if restaurant_id not in self.models:
            response.update({"reason": "Nuevo Restaurante", "model_type": "Fallback"})
            return response

        model = self.models[restaurant_id]
        le_customer = self.encoders[restaurant_id]['customer']
        le_item = self.encoders[restaurant_id]['item']
        top_sellers = self.top_sellers.get(restaurant_id, ["Plato Popular"])

        # ESTRATEGIA 1: PEDIDO RECURRENTE (Adaptive Bundle)
        try:
            recurrent_bundle = self.get_recurrent_bundle(restaurant_id, customer_clean)
            if recurrent_bundle:
                response.update({
                    "recommendation": recurrent_bundle,
                    "reason": f"Es tu pedido habitual en {restaurant_id}.",
                    "model_type": "Pattern Recognition (Recurrent Bundle)"
                })
                return response
        except Exception as e:
            print(f"Error en Bundle Logic: {e}")

        # ESTRATEGIA 2: MODELO (Top 3 Items)
        # Verificar si el cliente existe (usando ID limpio)
        if customer_clean not in le_customer.classes_:
            # COLD START: Devolver Top Sellers del Restaurante
            response.update({
                "recommendation": top_sellers, 
                "reason": f"Sugerencias más populares de {restaurant_id} (Cliente Nuevo)", 
                "model_type": "Heuristic (Top Sellers)"
            })
            return response

        try:
            customer_code = le_customer.transform([customer_clean])[0]
            probs = model.predict_proba([[customer_code, float(current_ticket_avg), int(hour), int(day)]])[0]
            
            top_3_indices = probs.argsort()[-3:][::-1]
            top_3_items = le_item.inverse_transform(top_3_indices)
            
            response.update({
                "recommendation": list(top_3_items),
                "reason": f"Basado en tus gustos variados en {restaurant_id}.",
                "model_type": "Neural Network (Top-3 Probabilistic)"
            })
            return response
        except Exception as e:
            # Error Fallback -> Top Sellers
            response.update({
                "recommendation": top_sellers, 
                "reason": f"Error calculando predicción (Fallback): {str(e)}", 
                "model_type": "Error Fallback"
            })
            return response

recommender = RestaurantRecommender()
df = load_data()
if not df.empty and 'restaurant_id' in df.columns:
    recommender.train(df)
else:
    print("Advertencia: DataFrame vacío.")
