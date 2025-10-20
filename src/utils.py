import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV


def load_and_clean_data(file_path):
    """
    Carga los datos y realiza la limpieza básica (elimina filas con nulos en columnas clave).
    """
    print(f"Cargando datos desde: {file_path}")
    data = pd.read_csv(file_path)

    # Columnas que usaremos en el modelo (incluye el target)
    features_for_model = [
        'diabetes_prevalence',
        'Percent of Population Aged 60+',
        'MEDHHINC_2018',
        'PCTPOVALL_2018',
        'Percent of adults with a bachelor\'s degree or higher 2014-18',
        'Unemployment_rate_2018',
        'Urban_rural_code',
        'Active Physicians per 100000 Population 2018 (AAMC)',
    ]

    # Limpieza: Eliminar filas con NaN en las columnas seleccionadas
    # (Estrategia conservadora para un modelo final)
    data_cleaned = data[features_for_model].dropna()

    print(f"Datos limpios: {data_cleaned.shape[0]} filas restantes.")
    return data_cleaned


def create_model_pipeline(X_train):
    """
    Define y entrena un Pipeline para el preprocesamiento y el modelo Lasso.
    """
    # 1. Definir los pasos del Pipeline
    # El escalador debe estar dentro del Pipeline para aplicarse correctamente
    # a las nuevas instancias de predicción.

    # 2. Definir el modelo (LassoCV para encontrar el mejor alpha automáticamente)
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 2, 100),
        cv=5,
        random_state=42,
        max_iter=10000
    )

    # 3. Construir el Pipeline: Escalar -> Modelo
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso_model', lasso_cv)
    ])

    return model_pipeline


if __name__ == '__main__':
    # Ejemplo de uso (esto NO se ejecuta al importar el módulo)
    # Debes asegurar que el archivo exista en la ruta 'data/raw/'
    try:
        df = load_and_clean_data('../data/raw/demographic_health_data.csv')
        df.to_csv('../data/processed/demographic_health_processed.csv', index=False)
        print("Dataset procesado guardado en data/processed/")
    except FileNotFoundError:
        print("No se puede ejecutar utils.py directamente sin el archivo en la ruta esperada.")