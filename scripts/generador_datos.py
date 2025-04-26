import random
import pandas as pd
from faker import Faker

def generar_datos_recepcion_optimizado(cantidad):
    # Tipos de muestras y urgencia
    tipos_muestras = [
        "Sangre", "Orina", "Materia fecal", 
        "Esputo", "LCR", "Secreción vaginal", "Secreción ótica"
    ]
    urgencias_muestras = ["Urgente", "Regular", "Especial"]
    faker = Faker()
    datos = []

    for _ in range(cantidad):
        fecha_hora = faker.date_time_between(start_date="-1y", end_date="now").strftime("%Y-%m-%d %H:%M")
        dia_semana = pd.to_datetime(fecha_hora).dayofweek  # 0 = lunes, 6 = domingo
        hora = pd.to_datetime(fecha_hora).hour
        volumen_muestras = random.randint(5, 20)  # Muestras recibidas en ese momento
        personal_disponible = random.randint(1, 5)  # Técnicos disponibles en el turno
        tipo_muestra = random.choice(tipos_muestras)  # Tipo específico de muestra
        urgencia_muestra = random.choice(urgencias_muestras)  # Urgencia de la muestra

        datos.append([fecha_hora, dia_semana, hora, volumen_muestras, personal_disponible, tipo_muestra, urgencia_muestra])

    # Estructura del DataFrame
    return pd.DataFrame(datos, columns=[
        "Fecha y Hora", "Dia_Semana", "Hora", "Volumen de Muestras", "Personal Disponible", "Tipo de Muestra", "Urgencia de Muestra"
    ])

if __name__ == "__main__":
    # Generar datos ficticios (puedes ajustar la cantidad)
    datos_ficticios = generar_datos_recepcion_optimizado(5000)

    # Guardar en archivo CSV
    datos_ficticios.to_csv("data/datos_recepcion_optimizado.csv", index=False)
    print("Datos ficticios generados y guardados en 'data/datos_recepcion_optimizado.csv'.")
    print(datos_ficticios.head(10))  # Muestra los primeros 10 registros
