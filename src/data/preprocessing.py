import numpy as np
import pandas as pd

class PINNPreprocessor:
    """
    Motor matemático para el preprocesamiento de variables de la PINN.
    Implementa la adimensionalización estricta para evitar gradientes patológicos
    durante la optimización del problema inverso (Advección-Difusión-Reacción).
    """
    def __init__(self, spatial_bounds: tuple, time_max: float, conc_max: float):

        #? spatial bounds: establece un rectángulo geográfico de referencia para escalar latitudes y longitudes.
        #? time_max: define el tiempo máximo esperado para escalar la variable temporal t a [0, 1].
        #? conc_max: define la concentración máxima esperada de PM2.5/PM10 para escalar la variable de concentración a [0, 1]. Valores negativos se mapearán a 0.
        """
        :param spatial_bounds: (min_lat, max_lat, min_lon, max_lon)
        :param time_max: tiempo máximo en segundos (o horas) para escalar t a [0, 1]
        :param conc_max: concentración máxima esperada (ug/m3) para PM2.5/PM10
        """
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = spatial_bounds
        self.time_max = time_max
        self.conc_max = conc_max

    def scale_spatial(self, lat: float, lon: float) -> tuple: 

        #? Las coordenadas geográficas se adimensionalizan primero a [0, 1] usando un escalamiento Min-Max basado en los límites definidos.
        #? Luego se mapean a [-1, 1] para mejorar la estabilidad numérica durante el entrenamiento de la PINN. 
        #? Esto es crucial para evitar que las diferencias de escala entre latitudes y longitudes causen problemas de convergencia en la optimización del modelo.
        """
        Adimensionaliza coordenadas geográficas al rango [-1, 1].
        """

        #? Lo ahce entre -1 y 1 porque las funciones de activación comunes en redes neuronales (ReLU, tanh) funcionan mejor con entradas centradas alrededor de cero.

        # Escalamiento Min-Max a [0, 1]
        lat_norm = (lat - self.min_lat) / (self.max_lat - self.min_lat)
        lon_norm = (lon - self.min_lon) / (self.max_lon - self.min_lon)
        
        # Mapeo a [-1, 1]
        x_scaled = 2.0 * lon_norm - 1.0
        y_scaled = 2.0 * lat_norm - 1.0
        
        return x_scaled, y_scaled

    def scale_time(self, t: float) -> float:

        #? El tiempo se adimensionaliza dividiendo por un valor máximo esperado (time_max). 
        #? Esto es esencial para que la variable temporal tenga una escala similar a las variables espaciales y de concentración.
        #? Facilita la optimización de la PINN.
        """
        Adimensionaliza el tiempo al rango [0, 1].
        """
        return t / self.time_max

    def scale_concentration(self, c: float) -> float:

        #? Se aplica un filtro de ruido para valores negativos; cuando un sensor se descalibra, puede reportar valores absurdos (ej. -10 ug/m3 o 9999 ug/m3).
        #? Estos valores se mapean a 0 para evitar que el modelo aprenda patrones erróneos.
        #? Luego, se escala la concentración dividiendo por un valor máximo esperado (conc_max) y limitando el resultado a 1. 
        """
        Escala la concentración de material particulado a [0, 1].
        Retorna 0 si es negativo (filtro de ruido).
        """
        if c < 0:
             return 0.0
        return min(c / self.conc_max, 1.0)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        #? El ensamblado de estas funciones de escalamiento se realiza en el método process_dataframe, que toma un DataFrame crudo con columnas 'latitud', 'longitud', 'timestamp' y 'pm25'. 
        #? Devuelve un DataFrame con columnas adicionales 'x_scaled', 'y_scaled', 't_scaled' y 'u_scaled' que contienen las versiones adimensionalizadas de las variables originales.
        """
        Toma un DataFrame crudo y aplica el escalamiento matemático.
        Espera columnas: 'latitud', 'longitud', 'timestamp', 'pm25'.
        """
        df_scaled = df.copy()
        
        # Eliminar NaNs
        df_scaled = df_scaled.dropna(subset=['latitud', 'longitud', 'timestamp', 'pm25'])
        
        # Aplicar escalamientos
        spatial_scaled = df_scaled.apply(
            lambda row: self.scale_spatial(row['latitud'], row['longitud']), axis=1
        )
        df_scaled['x_scaled'] = [s[0] for s in spatial_scaled]
        df_scaled['y_scaled'] = [s[1] for s in spatial_scaled]
        
        df_scaled['t_scaled'] = df_scaled['timestamp'].apply(self.scale_time)
        df_scaled['u_scaled'] = df_scaled['pm25'].apply(self.scale_concentration)
        
        return df_scaled
