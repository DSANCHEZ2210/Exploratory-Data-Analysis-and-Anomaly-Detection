# ==========================================
# LIBRERÍAS
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from datetime import timedelta

# ==========================================
# 1. CARGA Y PREPARACIÓN DE LOS DATOS
# ==========================================
df = pd.read_csv('data/serie_temporal_mes.csv', index_col='user_ts')

# Convertimos el índice a datetime
df.index = pd.to_datetime(df.index)

# Ordenamos el índice por fecha
df = df.sort_index()

# 🔥 🔥 🔥 Aquí seleccionas la variable que quieres analizar 🔥 🔥 🔥
serie = df["numberOfActivatedRadiators_CurrentPreformNeckFinishTemperature.0"]

# Revisión del rango de fechas y tamaño
print(f"Rango de fechas: {serie.index.min()} --> {serie.index.max()}")
print(f"Cantidad de registros: {len(serie)}")

# Resampleo opcional si quieres entrenar el modelo con menos granularidad
serie_minuto = serie.resample('30s').mean().dropna()
print(f"Cantidad de registros resampleados (30 segundos): {len(serie_minuto)}")

# ==========================================
# 2. ENTRENAMIENTO INICIAL DEL SARIMA (Primera ventana)
# ==========================================
# Fechas de entrenamiento iniciales
start_train_date = pd.Timestamp('2025-01-04 18:00:00+00:00')
end_train_date = start_train_date + timedelta(days=7)

# Datos de entrenamiento
train_data = serie[start_train_date:end_train_date]

# Modelo SARIMA (ajustado previamente)
final_model = SARIMAX(train_data,
                      order=(4, 1, 2),
                      seasonal_order=(0, 1, 2, 60),
                      enforce_stationarity=False,
                      enforce_invertibility=False)

final_result = final_model.fit(disp=False)
print(final_result.summary())

# ==========================================
# 3. ANÁLISIS DE RESIDUOS DEL MODELO ENTRENADO
# ==========================================
residuals = final_result.resid

plt.figure(figsize=(14,4))
plt.plot(residuals)
plt.title('Residuos del modelo')
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Histograma de los residuos')
plt.grid()
plt.show()

sm.qqplot(residuals, line='s')
plt.title('Q-Q plot de los residuos')
plt.grid()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16,4))
plot_acf(residuals, ax=ax[0])
plot_pacf(residuals, ax=ax[1])
plt.show()

# ==========================================
# 4. SLIDING WINDOW (Pronóstico y Evaluación)
# ==========================================
train_days = 7
predict_days = 2
num_iterations = 3  # Se mueve 2 días cada vez para cubrir 6 días

forecast_all = []
real_all = []

for i in range(num_iterations):
    # Fechas para cada ventana
    start_train_date = pd.Timestamp('2025-01-04 18:00:00+00:00') + timedelta(days=predict_days*i)
    end_train_date = start_train_date + timedelta(days=train_days)
    start_forecast_date = end_train_date + timedelta(seconds=1)
    end_forecast_date = start_forecast_date + timedelta(days=predict_days)

    print(f"\nVENTANA {i+1}:")
    print(f"Entrenamiento: {start_train_date} --> {end_train_date}")
    print(f"Predicción: {start_forecast_date} --> {end_forecast_date}")

    # Subconjunto para entrenamiento
    train_data = serie[start_train_date:end_train_date]

    # Entrenamiento SARIMA en cada ventana
    model = SARIMAX(train_data,
                    order=(4, 1, 2),
                    seasonal_order=(0, 1, 2, 60),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result = model.fit(disp=False)

    # Datos reales del forecast para comparar
    real_data = serie[start_forecast_date:end_forecast_date]
    steps = len(real_data)

    # Forecast
    pred_uc = result.get_forecast(steps=steps)
    pred_ci = pred_uc.conf_int()
    forecast_mean = pred_uc.predicted_mean

    # Igualamos el índice al real para poder graficar y comparar
    forecast_mean.index = real_data.index
    pred_ci.index = real_data.index

    # Guardamos
    forecast_all.append(forecast_mean)
    real_all.append(real_data)

    # Métricas
    mae = mean_absolute_error(real_data.values, forecast_mean.values)
    rmse = np.sqrt(mean_squared_error(real_data.values, forecast_mean.values))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot de la predicción
    plt.figure(figsize=(14,5))
    plt.plot(real_data.index, real_data.values, label='Serie Real (segundos)', alpha=0.5)
    plt.plot(forecast_mean.index, forecast_mean.values, color='red', label='Forecast SARIMA interpolado')
    plt.fill_between(forecast_mean.index,
                     pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title(f"Predicción SARIMA vs Datos Reales - Iteración {i+1}")
    plt.xlabel("Fecha")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# 5. EVALUACIÓN GLOBAL DE LAS 3 VENTANAS
# ==========================================
forecast_total = pd.concat(forecast_all)
real_total = pd.concat(real_all)

mae_total = mean_absolute_error(real_total.values, forecast_total.values)
rmse_total = np.sqrt(mean_squared_error(real_total.values, forecast_total.values))

print("\nEvaluación Global del Sliding Window:")
print(f"MAE Total: {mae_total:.4f}")
print(f"RMSE Total: {rmse_total:.4f}")

plt.figure(figsize=(16,6))
plt.plot(real_total.index, real_total.values, label='Serie Real', alpha=0.5)
plt.plot(forecast_total.index, forecast_total.values, color='red', label='Pronóstico SARIMA')
plt.title('Pronóstico Total SARIMA vs Datos Reales')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()