# api_ventas_ml/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from pathlib import Path # Importación clave para rutas robustas

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELOS
# ----------------------------------------------------------------------

# Determinar el directorio base donde reside main.py (la carpeta api_ventas_ml)
BASE_DIR = Path(__file__).resolve().parent

# Lista EXACTA de features de XGBoost obtenida de Google Colab
XGB_FEATURES = [
    'estoque', 'preco', 'Mes_2', 'Mes_3', 'Mes_4', 'Mes_5', 'Mes_6', 'Mes_7', 'Mes_8', 
    'Mes_9', 'Mes_10', 'Mes_11', 'Mes_12', 'Dia_Semana_1', 'Dia_Semana_2', 
    'Dia_Semana_3', 'Dia_Semana_4', 'Dia_Semana_5', 'Dia_Semana_6'
]

# Inicializar FastAPI
app = FastAPI()

# Cargar los modelos al iniciar la aplicación (ROBUSTO)
try:
    # Carga usando la ruta completa (BASE_DIR / nombre_archivo)
    rl_model = joblib.load(BASE_DIR / 'modelo_regresion_lineal.joblib')
    xgb_model = joblib.load(BASE_DIR / 'modelo_xgboost.joblib')
    print("Modelos cargados exitosamente.")
except Exception as e:
    # Manejo de error si los archivos no se encuentran o están corruptos
    print(f"Error al cargar modelos: {e}")
    rl_model = None
    xgb_model = None

# ----------------------------------------------------------------------
# 2. ESQUEMA DE DATOS DE ENTRADA
# ----------------------------------------------------------------------

# Definición del esquema de datos que la API espera recibir del portal web
class PredictionRequest(BaseModel):
    estoque: float
    preco: float
    Ano: int
    Mes: int
    Dia_Semana: int
    Es_Fin_Semana: int

# ----------------------------------------------------------------------
# 3. ENDPOINTS DE LA API
# ----------------------------------------------------------------------

# Endpoint de Testeo (Salud)
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Ventas activa."}

# Endpoint principal de predicción
@app.post("/predict_ventas")
def predict_ventas(data: PredictionRequest):
    if rl_model is None or xgb_model is None:
        return {"status": "error", "message": "Modelos no cargados en el servidor."}
        
    # Convertir el objeto de entrada a un DataFrame de Pandas
    input_data = data.model_dump()
    df_input = pd.DataFrame([input_data])

    # 1. Preparación de datos para REGRESIÓN LINEAL
    # El modelo RL solo necesita sus 6 columnas originales
    df_rl = df_input[['estoque', 'preco', 'Ano', 'Mes', 'Dia_Semana', 'Es_Fin_Semana']]
    
    # 2. Preparación de datos para XGBOOST (One-Hot Encoding y Alineación)
    
    # Codificar variables categóricas
    df_xgb = pd.get_dummies(df_input.copy(), columns=['Mes', 'Dia_Semana'], drop_first=True)
    
    # Asegurar que el DataFrame de entrada tenga las mismas columnas que el entrenamiento (XGB_FEATURES)
    # Rellenar con 0 si falta alguna columna (mes o día de la semana no visto)
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
