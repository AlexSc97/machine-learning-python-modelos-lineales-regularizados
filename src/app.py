import pandas as pd
from utils import load_and_clean_data, create_model_pipeline
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# Rutas de archivo
RAW_DATA_PATH = 'data/raw/demographic_health_data.csv'
MODEL_SAVE_PATH = 'models/final_model.pkl'

# Target y Features
TARGET_COL = 'diabetes_prevalence'


def train_and_save_model():
    """
    Flujo principal: carga, limpia, divide, entrena el modelo y lo guarda.
    """
    try:
        # 1. Cargar y limpiar datos
        df = load_and_clean_data(RAW_DATA_PATH)

        # 2. Separar X e y
        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]

        # 3. División en conjuntos de entrenamiento y prueba
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Crear y entrenar el Pipeline del modelo
        pipeline = create_model_pipeline(X_train)
        print("Iniciando entrenamiento del modelo LassoCV...")
        pipeline.fit(X_train, y_train)
        print("Entrenamiento completado.")

        # 5. Guardar el modelo entrenado
        with open(MODEL_SAVE_PATH, 'wb') as file:
            pickle.dump(pipeline, file)

        print(f"\n¡Modelo entrenado y guardado con éxito en {MODEL_SAVE_PATH}!")

    except FileNotFoundError:
        print(f"\nError: El archivo de datos no se encontró en la ruta: {RAW_DATA_PATH}")
    except Exception as e:
        print(f"\nOcurrió un error durante el entrenamiento: {e}")


if __name__ == '__main__':
    train_and_save_model()