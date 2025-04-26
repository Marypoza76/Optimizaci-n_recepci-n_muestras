import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# 1. Cargar datos
datos = pd.read_csv("data/datos_recepcion_optimizado.csv")
print(datos.head())  # Para verificar las primeras filas
print(datos.columns)  # Para listar todas las columnas disponibles

# Extraer la hora desde 'Fecha y Hora'
datos['Hora'] = pd.to_datetime(datos['Fecha y Hora']).dt.hour

# Crear nuevas variables a partir de los datos existentes
datos['Dia_Semana'] = pd.to_datetime(datos['Fecha y Hora']).dt.dayofweek  # Día de la semana (0 = lunes, 6 = domingo)
datos['Periodo_Dia'] = pd.cut(datos['Hora'], bins=[0, 6, 12, 18, 24], labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])  # Periodo del día
datos['Carga_Operativa'] = datos['Volumen de Muestras'] / datos['Personal Disponible']  # Relación entre volumen y personal disponible

# Simulación: Ajustar personal disponible y recalcular la carga operativa
datos['Escenario_Personal'] = datos['Personal Disponible'] + 1  # Incrementa personal
datos['Nueva_Carga_Operativa'] = datos['Volumen de Muestras'] / datos['Escenario_Personal']
print("Simulación de nuevo escenario de carga operativa:")
print(datos[['Hora', 'Carga_Operativa', 'Nueva_Carga_Operativa']].head(10))  # Muestra los primeros resultados del nuevo escenario

# Crear una nueva interacción entre volumen y personal disponible
datos['Interaccion_Volumen_Personal'] = datos['Volumen de Muestras'] * datos['Personal Disponible']

# Evaluar la calidad de los datos
print("Descripción de los datos:")
print(datos.describe())  # Estadísticas descriptivas de columnas numéricas

# Revisar las correlaciones entre las columnas numéricas
print("Correlaciones entre las columnas numéricas:")
print(datos.select_dtypes(include=["number"]).corr())  # Solo columnas numéricas

# Métricas agregadas por hora y día de la semana
print("Promedio de carga operativa por hora:")
print(datos.groupby('Hora')['Carga_Operativa'].mean())  # Promedio de carga por hora
print("Promedio de carga operativa por día de la semana:")
print(datos.groupby('Dia_Semana')['Carga_Operativa'].mean())  # Promedio de carga por día

# 2. Definir variables independientes y dependientes
X = pd.get_dummies(datos[["Personal Disponible", "Hora", "Tipo de Muestra", "Dia_Semana", "Periodo_Dia", "Interaccion_Volumen_Personal"]], drop_first=True)  # Convertir variables categóricas en dummies
y = datos["Carga_Operativa"]  # Variable objetivo

# Validación cruzada para Random Forest
print("Validación cruzada para Random Forest ajustado:")
modelo_rf_cv = RandomForestRegressor(
    n_estimators=200,  # Número de árboles
    max_depth=10,      # Máxima profundidad de los árboles
    min_samples_split=5,  # Tamaño mínimo para dividir nodos
    random_state=42
)
scores_rf = cross_val_score(modelo_rf_cv, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE promedio con validación cruzada (Random Forest): {-scores_rf.mean():.2f}")

# Validación cruzada para Gradient Boosting
print("Validación cruzada para Gradient Boosting:")
modelo_gb_cv = GradientBoostingRegressor(
    n_estimators=200,  # Número de estimadores
    learning_rate=0.1,  # Tasa de aprendizaje
    max_depth=3,       # Máxima profundidad
    random_state=42
)
scores_gb = cross_val_score(modelo_gb_cv, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"MAE promedio con validación cruzada (Gradient Boosting): {-scores_gb.mean():.2f}")

# 3. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Analizar estadísticas de los conjuntos de entrenamiento y prueba
print("Estadísticas del conjunto de entrenamiento:")
print(X_train.describe())
print("Estadísticas del conjunto de prueba:")
print(X_test.describe())

# 4. Modelo 1: Regresión Lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)
y_pred_lineal = modelo_lineal.predict(X_test)
error_lineal = mean_absolute_error(y_test, y_pred_lineal)
print(f"Error Absoluto Medio (Regresión Lineal): {error_lineal:.2f}")

# 5. Modelo 2: Random Forest ajustado
modelo_rf = RandomForestRegressor(
    n_estimators=100,  # Número de árboles
    max_depth=6,      # Máxima profundidad de los árboles
    min_samples_split=10,  # Tamaño mínimo para dividir nodos
    max_features='sqrt',  # Usar la raíz cuadrada del total de características
    random_state=42
)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)
error_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Error Absoluto Medio (Random Forest ajustado): {error_rf:.2f}")

# Analizar importancia de las características
importances = modelo_rf.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")

# 6. Modelo 3: Gradient Boosting
modelo_gb = GradientBoostingRegressor(
    n_estimators=200,  # Número de estimadores
    learning_rate=0.1,  # Tasa de aprendizaje
    max_depth=3,       # Máxima profundidad
    random_state=42
)
modelo_gb.fit(X_train, y_train)
y_pred_gb = modelo_gb.predict(X_test)
error_gb = mean_absolute_error(y_test, y_pred_gb)
print(f"Error Absoluto Medio (Gradient Boosting): {error_gb:.2f}")

# 7. Visualización: Regresión Lineal
plt.scatter(y_test, y_pred_lineal, alpha=0.7, color='blue', label="Lineal")
plt.title("Predicciones vs. Valores Reales (Regresión Lineal)")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.savefig("resultados_lineal.png")
plt.show()

# 8. Visualización: Random Forest
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green', label="Random Forest")
plt.title("Predicciones vs. Valores Reales (Random Forest)")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.savefig("resultados_rf.png")
plt.show()

# 9. Visualización: Gradient Boosting
plt.scatter(y_test, y_pred_gb, alpha=0.7, color='purple', label="Gradient Boosting")
plt.title("Predicciones vs. Valores Reales (Gradient Boosting)")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.savefig("resultados_gb.png")
plt.show()

# 9. Visualización: Gradient Boosting
plt.scatter(y_test, y_pred_gb, alpha=0.7, color='purple', label="Gradient Boosting")
plt.title("Predicciones vs. Valores Reales (Gradient Boosting)")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.savefig("resultados_gb.png")
plt.show()

# Gráfico: Carga Operativa por Hora
datos.groupby('Hora')['Carga_Operativa'].mean().plot(kind='bar', title='Carga Operativa por Hora')
plt.xlabel('Hora')
plt.ylabel('Carga Operativa Promedio')
plt.savefig("carga_operativa_hora.png")

# Gráfico: Carga Operativa por Día de la Semana
datos.groupby('Dia_Semana')['Carga_Operativa'].mean().plot(kind='bar', title='Carga Operativa por Día')
plt.xlabel('Día de la Semana')
plt.ylabel('Carga Operativa Promedio')
plt.savefig("carga_operativa_dia.png")
