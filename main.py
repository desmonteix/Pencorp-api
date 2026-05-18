from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import recommender

app = FastAPI(title="Restaurant AI Predictor — Hybrid CF + NCF")

class CustomerInput(BaseModel):
    restaurant_id: str
    customer_id: str
    ticket_average: float = 0.0
    is_new_customer: bool = False
    hour: int = 12
    day_of_week: int = 0

@app.get("/")
def read_root():
    return {"status": "AI Service Online (Hybrid: CF + NCF)"}

@app.post("/predict")
def predict_preference(input_data: CustomerInput):
    """
    Endpoint principal para n8n.
    Recibe datos del cliente y devuelve recomendación personalizada.
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

@app.get("/debug")
def debug_info():
    """
    Endpoint de diagnóstico.
    Muestra el estado del modelo, cuántos datos tiene cada restaurante,
    qué motor usa (CF o NCF), etc.
    """
    try:
        return recommender.get_debug_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain():
    """Fuerza un reentrenamiento del modelo con datos frescos de Supabase."""
    try:
        from model import load_data
        df = load_data()
        if df.empty:
            return {"status": "error", "message": "No se pudieron cargar datos."}
        recommender.train(df)
        return {"status": "ok", "message": "Modelo reentrenado.", "details": recommender.get_debug_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
