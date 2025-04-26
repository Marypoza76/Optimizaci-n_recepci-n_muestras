import pandas as pd

def cargar_datos(ruta_archivo):
    """Carga los datos del archivo CSV."""
    return pd.read_csv('/home/marillyn/optimizacion_recepcion_muestras/data/datos_muestras.csv')

def optimizar_asignacion(data):
    """
    Optimiza la asignación de personal y ordena las muestras
    para minimizar tiempos de espera.
    """
    # Ordenar las muestras por tiempo de procesamiento (prioridad más rápida primero)
    data = data.sort_values(by="Tiempo de Procesamiento (min)")

    # Asignar personal disponible proporcionalmente a la carga
    data["Personal Asignado"] = data["Tiempo de Procesamiento (min)"].apply(
        lambda x: min(5, max(1, int(x / 15)))  # Ejemplo: 1 técnico por cada 15 minutos
    )

    return data

def guardar_datos_optimizados(data, ruta_salida):
    """Guarda los datos optimizados en un archivo CSV."""
    data.to_csv(ruta_salida, index=False)

if __name__ == "__main__":
    # Ruta de entrada y salida de los datos
    ruta_entrada = "data/datos_muestras.csv"
    ruta_salida = "data/datos_optimizados.csv"

    # Cargar datos
    datos = cargar_datos(ruta_entrada)
    print("Datos cargados:\n", datos.head())

    # Optimizar asignación
    datos_optimizados = optimizar_asignacion(datos)
    print("\nDatos optimizados:\n", datos_optimizados.head())

    # Guardar datos optimizados
    guardar_datos_optimizados(datos_optimizados, ruta_salida)
    print(f"\nDatos optimizados guardados en {ruta_salida}.")
