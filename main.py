from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import recommender

app = FastAPI(title="Restaurant AI Predictor")

class CustomerInput(BaseModel):
    restaurant_id: str   # Identificador del Restaurante (ej. "Pizzeria_Don_Pepe")
    customer_id: str
    ticket_average: float = 0.0
    is_new_customer: bool = False
    hour: int = 12       # 0-23
    day_of_week: int = 0 # 0=Monday, 6=Sunday

@app.get("/")
def read_root():
    return {"status": "AI Service Online (Multi-Restaurant)"}

@app.post("/predict")
def predict_preference(input_data: CustomerInput):
    """
    Endpoint para n8n.
    Recibe: ID Restaurante, Datos Cliente, Contexto.
    """
    try:
        result = recommender.predict_recommendation(
            input_data.restaurant_id,
            input_data.customer_id, 
            input_data.ticket_average,
            input_data.hour,
            input_data.day_of_week
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
