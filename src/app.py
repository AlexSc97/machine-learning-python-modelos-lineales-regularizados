# Importo las funciones que prepare en utils
from utils import cargar_datos, preparar_datos, dividir_datos, guardar_graficos_comparacion
# Importo funciones y librerias necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
# Librerias para asegurarme de que existan los datos y no falle el codigo
import pickle
import os

from src.utils import cargar_datos, preparar_datos

# --- Creación de carpetas (si no existen) ---
# Buena práctica para asegurar que el script no falle si las carpetas no están creadas
os.makedirs('../models', exist_ok=True)
os.makedirs('../reports/figures', exist_ok=True)

# 1: Carga y preparación de datos
url_datos = 'https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv'
total_data = cargar_datos(url_datos)
total_data_processed = preparar_datos(total_data)
X_train, X_test, y_train, y_test = dividir_datos(total_data_processed, 'diabetes_prevalence')
# Guardo data procesada
total_data_processed.to_csv('../data/processed/total_data_processed.csv', index=False)

# Escalo las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2: Entrenamiento y evaluación del modelo linear simple
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred = linear_model.predict(X_test_scaled)
print(f"El error cuadratico medio es (RMSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"El coeficiente de determinación R² es: {r2_score(y_test, y_pred):.4f}")

# 3: Entrenamiento y evaluación del modelo linear regularizado Lasso y Ridge
lasso_model = Lasso(alpha=1.0, max_iter= 300)
lasso_model.fit(X_train_scaled, y_train)

# Predicciones con Lasso
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Entreno el modelo Ridge
ridge_model = Ridge(alpha=0.1, max_iter= 300)
ridge_model.fit(X_train_scaled, y_train)

# Predicciones con Ridge
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluación de los modelos
r2_lasso = r2_score(y_test, y_pred_lasso)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"R² del modelo Lasso: {r2_lasso:.4f}")
print(f"R² del modelo Ridge: {r2_ridge:.4f}")

# Error cuadratico medio
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"MSE del modelo Lasso: {mse_lasso:.4f}")
print(f"MSE del modelo Ridge: {mse_ridge:.4f}")

# 4: Visualización y comparación de rendimientos de los modelos
guardar_graficos_comparacion(y_test, y_pred_lasso, y_pred_ridge)

# 5: Guardando el mejor modelo
print(f"Guardando el mejor modelo (Ridge) en la carpeta 'models/...'")
pickle.dump(ridge_model, open('../models/ridge_model.pkl', 'wb'))
pickle.dump(linear_model, open('../models/linear_model.pkl', 'wb'))


