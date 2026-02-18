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
            
            # Asegurarse de tener hora y día (extraer de created_at si existe, sino simular)
            if 'created_at' in data.columns:
                data['created_at'] = pd.to_datetime(data['created_at'])
                data['hour_of_day'] = data['created_at'].dt.hour
                data['day_of_week'] = data['created_at'].dt.dayofweek
            else:
                # Si no hay fecha, no podemos entrenar contexto (usaremos fillers)
                data['hour_of_day'] = 12
                data['day_of_week'] = 0

            if not data.empty:
                print(f"Datos cargados de Supabase: {len(data)} registros.")
                return data
        except Exception as e:
            print(f"Error conectando a Supabase: {e}. Usando Mock Data.")
    
    print("Usando Datos Mock (Local)...")
    # Datos de entrenamiento Mock (Historial enriquecido)
    # 0 = Lunes, 6 = Domingo
    # Datos de entrenamiento Mock (Historial enriquecido)
    # 0 = Lunes, 6 = Domingo
    data = {
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C001', 'C003', 'C001'],
        'restaurant_id': ['RestA', 'RestA', 'RestA', 'RestA', 'RestA', 'RestA', 'RestA', 'RestA'],
        'order_item': ['Hamburguesa Doble', 'Ensalada Cesar', 'Pizza Pepperoni', 'Lomo Saltado', 'Alitas BBQ', 'Café Pasado', 'Gaseosa Inka Cola', 'Jugo de Naranja'],
        'ticket_value': [45.50, 32.00, 68.00, 55.00, 40.00, 8.00, 10.00, 12.00],
        'hour_of_day': [19, 13, 20, 14, 18, 8, 21, 9], # Hora militar
        'day_of_week': [4, 1, 5, 2, 6, 0, 5, 0] # 0=Lunes, 4=Viernes, 5=Sabado...
    }
    return pd.DataFrame(data)

class RestaurantRecommender:
    def __init__(self):
        # Diccionario de modelos: { 'Restaurante_A': Modelo_A, 'Restaurante_B': Modelo_B }
        self.models = {}
        self.encoders = {} # { 'Restaurante_A': { 'customer': le, 'item': le } }
        self.is_trained = False

    def train(self, df):
        # Entrenamos un modelo INDEPENDIENTE por cada restaurante
        restaurantes = df['restaurant_id'].unique()
        
        for rest_id in restaurantes:
            print(f"Entrenando modelo para: {rest_id}...")
            df_rest = df[df['restaurant_id'] == rest_id].copy()
            
            # Encoders locales (solo platos de este restaurante)
            le_customer = LabelEncoder()
            le_item = LabelEncoder()
            
            df_rest['customer_code'] = le_customer.fit_transform(df_rest['customer_id'])
            df_rest['item_code'] = le_item.fit_transform(df_rest['order_item'])
            
            # Features: Customer, Ticket (Global), Hora, Día
            X = df_rest[['customer_code', 'ticket_value', 'hour_of_day', 'day_of_week']]
            y = df_rest['item_code']
            
            # MLP Model
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
            model.fit(X, y)
            
            self.models[rest_id] = model
            self.encoders[rest_id] = { 'customer': le_customer, 'item': le_item }
            
        self.is_trained = True
        print("Todos los modelos de restaurante entrenados.")

    def predict_recommendation(self, restaurant_id, customer_id, current_ticket_avg=0, hour=12, day=0):
        if not self.is_trained:
            return "El modelo aún no está entrenado."
        
        # Verificar si existe modelo para este restaurante
        if restaurant_id not in self.models:
            return {
                "recommendation": "Plato del Día (Genérico)",
                "reason": f"Restaurante '{restaurant_id}' no tiene historial suficiente.",
                "model_type": "Fallback (New Restaurant)"
            }

        model = self.models[restaurant_id]
        le_customer = self.encoders[restaurant_id]['customer']
        le_item = self.encoders[restaurant_id]['item']

        # Lógica para Clientes Nuevos en ESTE restaurante (Cold Start Local)
        # Aunque tenga historial global (ticket_avg), en este local es nuevo.
        if customer_id not in le_customer.classes_:
            # Reglas simples basadas en la hora
            if hour < 11:
                return {
                    "recommendation": "Desayuno de la Casa", 
                    "reason": f"Cliente Nuevo en {restaurant_id} - Sugerencia de Mañana",
                    "model_type": "Heuristic (Time-based)"
                }
            elif 12 <= hour < 16:
                return {
                    "recommendation": "Menú Ejecutivo", 
                    "reason": f"Cliente Nuevo en {restaurant_id} - Sugerencia de Almuerzo",
                    "model_type": "Heuristic (Time-based)"
                }
            else:
                return {
                    "recommendation": "Especial del Chef", 
                    "reason": f"Cliente Nuevo en {restaurant_id} - Sugerencia de Noche",
                    "model_type": "Heuristic (Time-based)"
                }

        # Cliente Recurrente: Usamos la Red Neuronal de ESTE restaurante
        customer_code = le_customer.transform([customer_id])[0]
        prediction_code = model.predict([[customer_code, current_ticket_avg, hour, day]])
        recommended_item = le_item.inverse_transform(prediction_code)[0]
        
        return {
            "restaurant_id": restaurant_id,
            "customer_id": customer_id,
            "recommendation": recommended_item,
            "reason": f"Basado en tus pedidos anteriores en {restaurant_id}.",
            "model_type": "Neural Network (Contextual)"
        }

recommender = RestaurantRecommender()
df = load_data()
if not df.empty and 'restaurant_id' in df.columns:
    recommender.train(df)
else:
    print("Advertencia: DataFrame vacío o sin columna 'restaurant_id'.")
