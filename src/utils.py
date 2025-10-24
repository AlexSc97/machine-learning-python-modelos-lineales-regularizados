# Importo las librerias que se utilizaron en el EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Funciones para cargar datos, y preparar datos
def cargar_datos(url):
    """
    :param url: Carga datos desde una URL
    :return: Datos en formato CSV para su lectura con pandas
    """
    print("Cargando datos...")
    df = pd.read_csv(url)
    return df
def preparar_datos(df):
    """
    Limpia y prepara los datos para dejarlos utilizarlos en el modelo
    Elimina valores nulos y duplicados
    Selecciona las columnas más relevantes para el modelo
    """
    # Hago una copia del data frame original
    df_procesado = df.copy()
    # Elimino valores duplicados si existen
    if df_procesado.duplicated().sum() > 0:
        df_procesado.drop_duplicates(inplace=True)
    # Selecciono solo las columnas más relevantes basadas en el análisis previo
    data_model = df_procesado[[
        'diabetes_prevalence',
        'MEDHHINC_2018',
        'PCTPOVALL_2018',
        'Percent of adults with a bachelor\'s degree or higher 2014-18',
        'Unemployment_rate_2018',
        'Active Physicians per 100000 Population 2018 (AAMC)'
    ]].copy()
    return data_model
def dividir_datos(df, variable_objetivo):
    """

    :param df: Data Frame con columnas mas relevantes y sin nulos
    :param variable_objetivo: y
    :return: Regresa la data dividida en train y test separando la variable objetivo (X_train, X_test, y_train, y_test)

    """
    # Divido los datos en características (X) y variable objetivo (y)
    print(f"Dividir datos entrenamiento y prueba")
    X = df.drop(variable_objetivo, axis=1)
    y = df[variable_objetivo]
    # Divido en conjunto de entrenamiento y prueba (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def guardar_graficos_comparacion(y_test, y_pred_lasso, y_pred_ridge):
    """
    Genera y guarda los graficos finales de regresión para comparar el rendimiento del modelo antes y despues de optimización

    """
    # Grafico para modelo menos optimo Lasso
    print(f"Generando graficos...")
    plt.figure(figsize=(10,7))
    plt.subplot(1,2,1)
    sns.scatterplot(x= y_test, y=y_pred_lasso)
    plt.plot([0,28], [0,28], '--r', linewidth=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title("Valores reales vs predicciones del modelo")
    plt.grid(True)

    # Grafico para modelo optimizado Ridge
    plt.subplot(1,2,2)
    sns.scatterplot(x= y_test, y=y_pred_ridge)
    plt.plot([0,28], [0,28], '--r', linewidth=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones modelo Ridge')
    plt.title('Valores reales vs predicciones del modelo Ridge')
    plt.grid(True)

    # Guardar graficos
    plt.savefig('../reports/figures/comparacion_modelos.png')
    print("Gráficos guardados en 'reports/figures/comparacion_modelos.png'")

