import pandas as pd
import random

# Configuración de datos ficticios
tipos_muestra = ["Sangre", "Orina", "Tejido"]
personal_disponible = [1, 2, 3, 4, 5]
horas = [8, 10, 12, 14, 16, 18]
tiempos_procesamiento = [random.randint(15, 60) for _ in range(50)]

# Crear DataFrame
datos_ficticios = pd.DataFrame({
    "Tipo de Muestra": [random.choice(tipos_muestra) for _ in range(50)],
    "Personal Disponible": [random.choice(personal_disponible) for _ in range(50)],
    "Hora": [random.choice(horas) for _ in range(50)],
    "Tiempo de Procesamiento (min)": tiempos_procesamiento
})

# Guardar como CSV
datos_ficticios.to_csv("datos_optimizados.csv", index=False)

print("Archivo 'datos_optimizados.csv' creado con éxito.")
