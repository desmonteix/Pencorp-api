import requests
import json
import os

# Cargar datos de prueba
with open(r"C:\Users\matid\.gemini\antigravity\brain\4ddbbad8-5e9d-4204-b75a-a287645a8e83\test_data.json", "r") as f:
    test_data = json.load(f)

url = "http://localhost:8000/predict"

print("--- Iniciando Test del Modelo Predictivo ---")

for i, data in enumerate(test_data):
    print(f"\nCaso {i+1}: {data['customer_name']} ({data['customer_id']})")
    
    # Preparamos el payload para la API
    # Asumimos que si el ID está en nuestra "base de datos mock" (C001-C005 fueron usados para test_data, 
    # pero en model.py el mock tiene C001, C002, C003, C004, C005?
    # Revisemos model.py:
    # IDs en entrenamiento: C001, C002, C003, C004, C005, C001, C003.
    # Entonces todos los del json deberían ser 'conocidos' excepto si forzamos uno nuevo.
    
    payload = {
        "customer_id": data["customer_id"],
        "ticket_average": data["ticket_average"],
        "is_new_customer": False,
        "hour": 20, # Simulamos cena
        "day_of_week": 5 # Sábado
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"   Input: Ticket ${data['ticket_average']} | Hora: 20:00 (Sáb)")
            print(f"   Recomendación IA: {result['recommendation']}")
            print(f"   Razón: {result['reason']}")
        else:
            print(f"   Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   Error de conexión: {e}")

print("\n--- Test Cliente Nuevo (Contexto Temporal) ---")
# Caso A: Mañana (Desayuno)
payload_morning = {
    "customer_id": "NEW999",
    "ticket_average": 15.0,
    "is_new_customer": True,
    "hour": 9,
    "day_of_week": 0
}
# Caso B: Noche (Cena)
payload_night = {
    "customer_id": "NEW998",
    "ticket_average": 50.0,
    "is_new_customer": True,
    "hour": 21,
    "day_of_week": 5
}

try:
    print(f"1. Nuevo Cliente a las 9:00 AM:")
    resp1 = requests.post(url, json=payload_morning).json()
    print(f"   Recomendación: {resp1['recommendation']}")

    print(f"2. Nuevo Cliente a las 9:00 PM:")
    resp2 = requests.post(url, json=payload_night).json()
    print(f"   Recomendación: {resp2['recommendation']}")
except Exception as e:
    print(e)
