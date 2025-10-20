# main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# --- 1. Carga de Modelos y Preparación ---

# Cargar los modelos al iniciar la aplicación
try:
    rl_model = joblib.load('modelo_regresion_lineal.joblib')
    xgb_model = joblib.load('modelo_xgboost.joblib')
    print("Modelos cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    rl_model = None
    xgb_model = None

app = FastAPI()

# Definir la estructura de datos que la API espera recibir
# Se usan las 6 variables predictoras del modelo Lineal y las One-Hot de XGBoost
class PredictionRequest(BaseModel):
    estoque: float
    preco: float
    Ano: int
    Mes: int
    Dia_Semana: int
    Es_Fin_Semana: int

# Obtener las columnas de entrenamiento de XGBoost (esto es crucial)
# Debes obtener esta lista de columnas desde tu notebook de Colab
# Ejemplo:
XGB_FEATURES = [
    'estoque', 'preco', 'Ano', 'Es_Fin_Semana', 
    'Mes_2', 'Mes_3', 'Mes_4', 'Mes_5', 'Mes_6', 'Mes_7', 'Mes_8', 'Mes_9', 'Mes_10', 'Mes_11', 'Mes_12', 
    'Dia_Semana_1', 'Dia_Semana_2', 'Dia_Semana_3', 'Dia_Semana_4', 'Dia_Semana_5', 'Dia_Semana_6'
]


# --- 2. Endpoint de Predicción ---

@app.post("/predict_ventas")
def predict_ventas(data: PredictionRequest):
    # Convertir el objeto de entrada a un DataFrame
    input_data = data.model_dump()
    df_input = pd.DataFrame([input_data])

    # 1. Preparación de datos para REGRESIÓN LINEAL
    # El modelo RL solo necesita sus 6 columnas originales
    df_rl = df_input[['estoque', 'preco', 'Ano', 'Mes', 'Dia_Semana', 'Es_Fin_Semana']]
    
    # 2. Preparación de datos para XGBOOST (One-Hot Encoding)
    # Codificar y alinear las columnas
    df_xgb = pd.get_dummies(df_input.copy(), columns=['Mes', 'Dia_Semana'], drop_first=True)
    df_xgb = df_xgb.reindex(columns=XGB_FEATURES, fill_value=0)
    
    # --- 3. Generar Predicciones ---
    
    rl_pred = rl_model.predict(df_rl)[0].round(2)
    xgb_pred = xgb_model.predict(df_xgb)[0].round(2)

    return {
        "status": "success",
        "valores_ingresados": input_data,
        "prediccion_regresion_lineal": float(rl_pred),
        "prediccion_xgboost": float(xgb_pred)
    }

# --- 4. Endpoint de Testeo (Salud) ---
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Ventas activa."}