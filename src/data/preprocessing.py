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
        self.min_elev = 1400.0 # Aproximación baja del río Medellín
        self.max_elev = 3000.0 # Aproximación de las montañas circundantes
        
        if len(spatial_bounds) > 4: # Por si el usuario pasa elev_bounds como kwargs luego
            pass

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

    def get_elevations(self, lat_lons: list) -> list:
        """
        Consulta la Open-Elevation API para obtener la topografía real de los sensores.
        Añade 2 metros asumiendo la altura estándar del poste del sensor.
        """
        import requests
        import logging
        
        # Batch request para no saturar la API
        locations_str = "|".join([f"{lat},{lon}" for lat, lon in lat_lons])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"
        
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                # Z = Elevación del terreno + 2 metros (altura del sensor)
                return [res['elevation'] + 2.0 for res in data['results']]
        except Exception as e:
            logging.error(f"Error consultando Open-Elevation: {e}")
            
        # Fallback: Si la API falla, asume una elevación plana promedio de 1500m
        logging.warning("Usando elevación fallback de 1500m.")
        return [1500.0] * len(lat_lons)

    def scale_elevation(self, elev: float) -> float:
        """Escala la elevación al rango [0, 1] para la coordenada z."""
        z_norm = (elev - self.min_elev) / (self.max_elev - self.min_elev)
        return max(0.0, min(z_norm, 1.0)) # Clamping a [0,1]

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
        
        # Manejo de DataFrame vacío para evitar errores de .apply en Pandas
        if df_scaled.empty:
            for col in ['x_scaled', 'y_scaled', 'elevacion_real', 'z_scaled', 't_scaled', 'u_scaled', 'T_scaled', 'x', 'y', 'z', 't', 'u', 'T']:
                df_scaled[col] = pd.Series(dtype='float64')
            return df_scaled
        
        # Aplicar escalamientos
        spatial_scaled = df_scaled.apply(
            lambda row: self.scale_spatial(row['latitud'], row['longitud']), axis=1
        )
        df_scaled['x_scaled'] = [s[0] for s in spatial_scaled]
        df_scaled['y_scaled'] = [s[1] for s in spatial_scaled]
        
        # Topografía (Eje Z) - Optimizado para coordenadas únicas
        unique_coords = df_scaled[['latitud', 'longitud']].drop_duplicates()
        lat_lons_unique = list(zip(unique_coords['latitud'], unique_coords['longitud']))
        
        # Consultar elevación solo de las 21 estaciones únicas
        elevations_unique = self.get_elevations(lat_lons_unique)
        
        # Crear diccionario de mapeo
        elev_dict = {coord: elev for coord, elev in zip(lat_lons_unique, elevations_unique)}
        
        df_scaled['elevacion_real'] = df_scaled.apply(lambda row: elev_dict[(row['latitud'], row['longitud'])], axis=1)
        df_scaled['z_scaled'] = df_scaled['elevacion_real'].apply(self.scale_elevation)
        
        df_scaled['t_scaled'] = df_scaled['timestamp'].apply(self.scale_time)
        df_scaled['u_scaled'] = df_scaled['pm25'].apply(self.scale_concentration)
        
        # Perfil de temperatura basado en la elevación (inversión térmica)
        # Mapea z_scaled ∈ [0, 1] → T ∈ [-1, 1] (consistente con BCs de Julia: T(z=0)=-1, T(z=1)=1)
        df_scaled['T_scaled'] = 2.0 * df_scaled['z_scaled'] - 1.0
        
        # Opcional: Estandarizar nombres para Julia si se desea guardar el JSON
        df_scaled['x'] = df_scaled['x_scaled']
        df_scaled['y'] = df_scaled['y_scaled']
        df_scaled['z'] = df_scaled['z_scaled']
        df_scaled['t'] = df_scaled['t_scaled']
        df_scaled['u'] = df_scaled['u_scaled']
        df_scaled['T'] = df_scaled['T_scaled']
        
        return df_scaled

    def process_wind_dataframe(self, df: pd.DataFrame, time_min_ref: float) -> pd.DataFrame:
        """
        Procesa el DataFrame de viento y calcula vx, vy adimensionales.
        """
        df_scaled = df.copy()
        
        # Eliminar NaNs en las columnas críticas
        df_scaled = df_scaled.dropna(subset=['latitud', 'longitud', 'timestamp', 'vx', 'vy'])
        
        if df_scaled.empty:
            for col in ['x_scaled', 'y_scaled', 'elevacion_real', 'z_scaled', 't_scaled', 'vx_scaled', 'vy_scaled', 'x', 'y', 'z', 't', 'vx', 'vy']:
                df_scaled[col] = pd.Series(dtype='float64')
            return df_scaled
            
        # Escalamiento espacial
        spatial_scaled = df_scaled.apply(
            lambda row: self.scale_spatial(row['latitud'], row['longitud']), axis=1
        )
        df_scaled['x_scaled'] = [s[0] for s in spatial_scaled]
        df_scaled['y_scaled'] = [s[1] for s in spatial_scaled]
        
        # Elevaciones
        unique_coords = df_scaled[['latitud', 'longitud']].drop_duplicates()
        lat_lons_unique = list(zip(unique_coords['latitud'], unique_coords['longitud']))
        elevations_unique = self.get_elevations(lat_lons_unique)
        elev_dict = {coord: elev for coord, elev in zip(lat_lons_unique, elevations_unique)}
        df_scaled['elevacion_real'] = df_scaled.apply(lambda row: elev_dict[(row['latitud'], row['longitud'])], axis=1)
        df_scaled['z_scaled'] = df_scaled['elevacion_real'].apply(self.scale_elevation)
        
        # Escalar tiempo relativo al mínimo de PM2.5 para mantener sincronización
        df_scaled['timestamp_rel'] = df_scaled['timestamp'] - time_min_ref
        df_scaled['t_scaled'] = df_scaled['timestamp_rel'].apply(self.scale_time)
        
        # Escalar velocidades (viento máx = 10 m/s)
        df_scaled['vx_scaled'] = df_scaled['vx'] / 10.0
        df_scaled['vy_scaled'] = df_scaled['vy'] / 10.0
        
        # Mapear a nombres de columnas que Julia espera
        df_scaled['x'] = df_scaled['x_scaled']
        df_scaled['y'] = df_scaled['y_scaled']
        df_scaled['z'] = df_scaled['z_scaled']
        df_scaled['t'] = df_scaled['t_scaled']
        df_scaled['vx'] = df_scaled['vx_scaled']
        df_scaled['vy'] = df_scaled['vy_scaled']
        
        return df_scaled

if __name__ == "__main__":
    import json
    import os
    print("Iniciando preprocesamiento dimensional (x, y, z, t)...")
    
    # 1. Cargar datos del scraper PM2.5
    try:
        df_raw = pd.read_json("datos_oficiales_pm25.json")
        print(f"Cargados {len(df_raw)} registros temporales de PM2.5.")
    except Exception as e:
        print("Error leyendo datos_oficiales_pm25.json:", e)
        exit(1)
        
    # Limites del Valle de Aburrá aprox
    spatial_bounds = (6.0, 6.45, -75.7, -75.3) # (min_lat, max_lat, min_lon, max_lon)
    
    time_min = df_raw['timestamp'].min() if not df_raw.empty else 0.0
    time_max = df_raw['timestamp'].max() - time_min if not df_raw.empty else 1.0
    
    # Ajustamos timestamp a relativo desde el minimo para que empiece en 0
    if not df_raw.empty:
        df_raw['timestamp'] = df_raw['timestamp'] - time_min
    
    conc_max = 100.0 # ug/m3 de pm25 máximo esperado
    
    preprocessor = PINNPreprocessor(spatial_bounds, time_max, conc_max)
    df_pinn = preprocessor.process_dataframe(df_raw)
    
    # Guardar en un nuevo archivo solo con las columnas procesadas para la PINN
    cols_to_save = ['id', 'x', 'y', 'z', 't', 'u', 'T', 'elevacion_real', 'pm25', 'latitud', 'longitud']
    df_final = df_pinn[cols_to_save].copy()
    
    output_file = "datos_siata_temporal.json"
    df_final.to_json(output_file, orient='records', indent=4)
    print(f"Preprocesamiento exitoso. Datos listos guardados en '{output_file}'.")
    
    # 2. Cargar y procesar datos de viento si existen
    if os.path.exists("datos_oficiales_viento.json"):
        try:
            df_wind_raw = pd.read_json("datos_oficiales_viento.json")
            print(f"Cargados {len(df_wind_raw)} registros de viento.")
            
            df_wind_processed = preprocessor.process_wind_dataframe(df_wind_raw, time_min)
            
            cols_wind = ['id', 'x', 'y', 'z', 't', 'vx', 'vy', 'elevacion_real', 'latitud', 'longitud']
            df_wind_final = df_wind_processed[cols_wind].copy()
            
            wind_output = "datos_meteorologicos_viento.json"
            df_wind_final.to_json(wind_output, orient='records', indent=4)
            print(f"Preprocesamiento de viento exitoso. Guardado en '{wind_output}'.")
        except Exception as e:
            print("Error preprocesando datos de viento:", e)
    else:
        print("[WARN] datos_oficiales_viento.json no encontrado (portal down). Generando datos_meteorologicos_viento.json espacial y topográficamente consistentes para entrenamiento...")
        try:
            # Obtener estaciones únicas de pm25 para ubicar los sensores meteorológicos simulados en los mismos puntos reales
            unique_stations = df_final[['id', 'x', 'y', 'z', 'elevacion_real', 'latitud', 'longitud']].drop_duplicates()
            
            simulated_winds = []
            import numpy as np
            # Generar perfiles de viento para 5 instantes de tiempo para asimilación dinámica
            for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for _, row in unique_stations.iterrows():
                    x_c = row['x']
                    y_c = row['y']
                    z_c = row['z']
                    
                    # Dinámica de laderas (Vientos anabáticos/catabáticos):
                    # El viento asciende por las laderas durante el día (t ~ 0.5) y desciende de noche
                    # Transversal (vx): influenciado por la ladera x y la altitud z
                    vx_sim = -0.15 * x_c * np.cos(2 * np.pi * t_val) - 0.05 * z_c
                    # Longitudinal (vy): flujo constante canalizado de sur a norte (de -y a +y) con ciclo diurno
                    vy_sim = 0.20 + 0.10 * np.sin(2 * np.pi * t_val)
                    
                    simulated_winds.append({
                        'id': f"W-{row['id']}",
                        'x': x_c,
                        'y': y_c,
                        'z': z_c,
                        't': t_val,
                        'vx': vx_sim,
                        'vy': vy_sim,
                        'elevacion_real': row['elevacion_real'],
                        'latitud': row['latitud'],
                        'longitud': row['longitud']
                    })
            
            df_wind_sim = pd.DataFrame(simulated_winds)
            wind_output = "datos_meteorologicos_viento.json"
            df_wind_sim.to_json(wind_output, orient='records', indent=4)
            print(f"[OK] Generación de viento simulado exitosa. {len(df_wind_sim)} registros guardados en '{wind_output}'.")
        except Exception as e:
            print("Error generando viento simulado:", e)



