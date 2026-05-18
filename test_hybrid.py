"""
Test del modelo híbrido con datos simulados.
Genera 10 pedidos falsos, entrena el modelo, verifica que las
recomendaciones sean distintas por cliente, y luego limpia todo.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importamos SOLO las clases, NO el módulo completo (para no trigger load_data)
import sys
sys.modules.pop('model', None)  # Limpiar cache si existe

# Importar clases directamente
from model import CollaborativeFilter, HybridRecommender

print("=" * 60)
print("TEST DEL MODELO HÍBRIDO — Datos Simulados")
print("=" * 60)

# ============================================================
# PASO 1: Crear 10 pedidos falsos (ya en formato "explotado")
# ============================================================
# Simulamos 3 clientes con gustos DISTINTOS para verificar
# que el modelo recomiende cosas diferentes a cada uno.
#
# Cliente A (Carlos): Ama las pizzas
# Cliente B (María):  Ama las empanadas
# Cliente C (José):   Mezcla de todo

now = datetime.now()
mock_orders = [
    # Carlos: 4 pedidos, mayormente pizzas
    {"customer_id_clean": "51999111222", "order_item": "Pizza Mozzarella Grande",   "ticket_value": 45.0, "hour_of_day": 20, "day_of_week": 5, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=10), "bundle_signature": "Pizza Mozzarella Grande"},
    {"customer_id_clean": "51999111222", "order_item": "Pizza Americana Grande",    "ticket_value": 48.0, "hour_of_day": 21, "day_of_week": 6, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=7), "bundle_signature": "Pizza Americana Grande"},
    {"customer_id_clean": "51999111222", "order_item": "Pizza Napolitana Grande",   "ticket_value": 48.0, "hour_of_day": 20, "day_of_week": 4, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=5), "bundle_signature": "Pizza Napolitana Grande"},
    {"customer_id_clean": "51999111222", "order_item": "Coca Cola",                 "ticket_value": 8.0,  "hour_of_day": 20, "day_of_week": 5, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=3), "bundle_signature": "Coca Cola"},

    # María: 3 pedidos, mayormente empanadas
    {"customer_id_clean": "51999333444", "order_item": "Empanada de Carne",         "ticket_value": 12.0, "hour_of_day": 13, "day_of_week": 2, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=8), "bundle_signature": "Empanada de Carne"},
    {"customer_id_clean": "51999333444", "order_item": "Empanada de Jamón y Queso", "ticket_value": 12.0, "hour_of_day": 14, "day_of_week": 3, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=6), "bundle_signature": "Empanada de Jamón y Queso"},
    {"customer_id_clean": "51999333444", "order_item": "Empanada de Carne",         "ticket_value": 12.0, "hour_of_day": 13, "day_of_week": 5, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=2), "bundle_signature": "Empanada de Carne"},

    # José: 3 pedidos, variado
    {"customer_id_clean": "51999555666", "order_item": "Milanesa Napolitana",       "ticket_value": 35.0, "hour_of_day": 21, "day_of_week": 6, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=9), "bundle_signature": "Milanesa Napolitana"},
    {"customer_id_clean": "51999555666", "order_item": "Pizza Mozzarella Grande",   "ticket_value": 45.0, "hour_of_day": 20, "day_of_week": 4, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=4), "bundle_signature": "Pizza Mozzarella Grande"},
    {"customer_id_clean": "51999555666", "order_item": "Coca Cola",                 "ticket_value": 8.0,  "hour_of_day": 20, "day_of_week": 5, "restaurant_id": "Patragonia_Test", "created_at": now - timedelta(days=1), "bundle_signature": "Coca Cola"},
]

df = pd.DataFrame(mock_orders)
df['created_at'] = pd.to_datetime(df['created_at'], utc=True)

print(f"\n📊 Datos generados: {len(df)} registros")
print(f"   Clientes: {df['customer_id_clean'].nunique()}")
print(f"   Productos: {df['order_item'].nunique()}")
print(f"   Restaurante: {df['restaurant_id'].unique()[0]}")

# ============================================================
# PASO 2: Entrenar el modelo con los datos falsos
# ============================================================
print("\n🔧 Entrenando modelo...")
test_recommender = HybridRecommender()
test_recommender.train(df)

# ============================================================
# PASO 3: Probar predicciones para cada cliente
# ============================================================
clientes = [
    {"id": "51999111222", "nombre": "Carlos (amante de pizzas)", "ticket": 45.0},
    {"id": "51999333444", "nombre": "María (amante de empanadas)", "ticket": 12.0},
    {"id": "51999555666", "nombre": "José (variado)", "ticket": 30.0},
    {"id": "51999999999", "nombre": "NUEVO (nunca ha pedido)", "ticket": 0.0},
]

print("\n" + "=" * 60)
print("📋 RESULTADOS DE PREDICCIÓN")
print("=" * 60)

results = {}
for c in clientes:
    res = test_recommender.predict_recommendation(
        restaurant_id="Patragonia_Test",
        customer_id=c["id"],
        current_ticket_avg=c["ticket"],
        hour=20,
        day=5
    )
    results[c["nombre"]] = res["recommendation"]
    print(f"\n👤 {c['nombre']}")
    print(f"   Recomendación: {res['recommendation']}")
    print(f"   Razón: {res['reason']}")
    print(f"   Motor: {res['model_type']}")
    if res.get('model_details', {}).get('user_profile'):
        profile = res['model_details']['user_profile']
        if isinstance(profile, dict):
            print(f"   Pedidos: {profile.get('total_orders', '?')} | Ticket: ${profile.get('avg_ticket', '?')} | Favoritos: {profile.get('favorite_items', [])}")

# ============================================================
# PASO 4: Verificar que las recomendaciones son DISTINTAS
# ============================================================
print("\n" + "=" * 60)
print("✅ VERIFICACIÓN DE PERSONALIZACIÓN")
print("=" * 60)

carlos = results["Carlos (amante de pizzas)"]
maria = results["María (amante de empanadas)"]
jose = results["José (variado)"]

# Carlos debería tener pizzas en sus recomendaciones
carlos_has_pizza = any("Pizza" in item for item in carlos)
# María debería tener empanadas
maria_has_empanada = any("Empanada" in item for item in maria)
# Las recomendaciones NO deberían ser idénticas entre clientes
all_different = carlos != maria

print(f"   Carlos recibe pizzas:      {'✅ SÍ' if carlos_has_pizza else '❌ NO'} → {carlos}")
print(f"   María recibe empanadas:    {'✅ SÍ' if maria_has_empanada else '❌ NO'} → {maria}")
print(f"   Recomendaciones distintas: {'✅ SÍ' if all_different else '❌ NO'}")

if carlos_has_pizza and maria_has_empanada and all_different:
    print("\n🎉 ¡TEST EXITOSO! El modelo personaliza correctamente por cliente.")
else:
    print("\n⚠️ El modelo necesita ajustes — las recomendaciones no son suficientemente distintas.")

# ============================================================
# PASO 5: Probar endpoint /debug
# ============================================================
print("\n" + "=" * 60)
print("🔍 DEBUG INFO")
print("=" * 60)
debug = test_recommender.get_debug_info()
for rid, stats in debug["restaurants"].items():
    print(f"   Restaurante: {rid}")
    print(f"   Motor: {stats['engine']} | Registros: {stats['total_records']} | Usuarios: {stats['unique_users']} | Items: {stats['unique_items']}")
    print(f"   NCF activo: {'SÍ' if stats.get('has_ncf') else 'NO (menos de 1000 registros)'}")

# ============================================================
# PASO 6: LIMPIEZA — Eliminar datos de prueba de memoria
# ============================================================
del test_recommender
del df
del mock_orders
print("\n🧹 Datos de prueba eliminados de memoria.")
print("   El modelo de producción (Supabase) NO fue afectado.")
print("=" * 60)
