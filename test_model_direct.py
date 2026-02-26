import json
from model import recommender, load_data

print("Cargando datos...")
df = load_data()
if not df.empty:
    print("Entrenando...")
    recommender.train(df)

customers = [
    {"id": "C001", "name": "Carlos Gomez", "ticket": 25.0, "hour": 20, "day": 5},
    {"id": "C002", "name": "Maria Lopez", "ticket": 45.0, "hour": 20, "day": 5},
    {"id": "C003", "name": "Jose Perez", "ticket": 12.0, "hour": 20, "day": 5},
    {"id": "C004", "name": "Ana Martinez", "ticket": 60.0, "hour": 20, "day": 5},
    {"id": "C005", "name": "Luis Torres", "ticket": 8.0, "hour": 20, "day": 5},
]

print("\n--- Resultados de Predicción ---")
rest_id = "Restaurante_A"
for c in customers:
    res = recommender.predict_recommendation(
        restaurant_id=rest_id,
        customer_id=c['id'],
        current_ticket_avg=c['ticket'],
        hour=c['hour'],
        day=c['day']
    )
    print(f"Cliente: {c['name']} | Ticket: ${c['ticket']} -> Sugerencia: {res.get('recommendation')}")
