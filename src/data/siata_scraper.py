import requests
import pandas as pd
import logging
import json
from typing import Optional, List, Dict
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class SiataOfficialNetwork:
    """
    Cliente para la ingesta de datos de la Red Oficial de Calidad de Aire y Meteorología del SIATA.
    Se separa la calidad de aire del viento, dado que muy pocos sensores miden ambas cosas al tiempo.
    """

    def __init__(self):
        # Endpoints de Datos Abiertos
        self.endpoint_pm25 = "https://datosabiertos.metropol.gov.co/sites/default/files/uploaded_resources/Datos_SIATA_Aire_pm25.json"
        self.endpoint_wind = "https://datosabiertos.metropol.gov.co/sites/default/files/uploaded_resources/Datos_SIATA_Vaisala_viento.json"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SIATA-Research-Agent/2.0',
            'Accept': 'application/json'
        })
        # Deshabilitar advertencias de SSL rotos típicos de dominios gubernamentales
        import urllib3
        urllib3.disable_warnings()

    def fetch_data(self, url: str) -> Optional[List[Dict]]:
        """Descarga JSON en tiempo real desde un endpoint específico."""
        try:
            # verify=False porque SIATA frecuentemente tiene problemas con el certificado SSL
            response = self.session.get(url, timeout=15, verify=False)
            response.raise_for_status()
            data = response.json()
            if not data:
                return None
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error HTTP al contactar {url}: {e}")
            return None

    def filter_pm25(self, df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
        """Filtro estricto para red oficial (aislando anomalías extremas)."""
        if df.empty or 'pm25' not in df.columns:
            return df
            
        df_clean = df[(df['pm25'] >= 0) & (df['pm25'] < 500)].copy()
        
        if len(df_clean) < 5:
            return df_clean
            
        clf = IsolationForest(contamination=contamination, random_state=42)
        X = df_clean['pm25'].to_numpy().reshape(-1, 1)
        preds = clf.fit_predict(X)
        return df_clean[preds == 1]

    def process_pm25_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        parsed = []
        for st in raw_data:
            lat = float(st.get('latitud', 0))
            lon = float(st.get('longitud', 0))
            st_id = st.get('codigoSerial', st.get('id', 'unknown'))
            
            # El JSON de Datos Abiertos tiene un arreglo "datos" con la serie de tiempo
            datos_historicos = st.get('datos', [])
            
            if not datos_historicos:
                # Fallback si no tiene arreglo "datos" (formato antiguo)
                try:
                    parsed.append({
                        'id': st_id,
                        'latitud': lat,
                        'longitud': lon,
                        'pm25': float(st.get('pm25', 0)),
                        'timestamp': st.get('fecha_hora', 0)
                    })
                except (ValueError, TypeError):
                    pass
                continue
                
            for dp in datos_historicos:
                try:
                    import datetime
                    # Convertir la fecha "YYYY-MM-DD HH:MM:SS" a timestamp unix
                    fecha_str = dp.get('fecha', '')
                    if fecha_str:
                        dt = datetime.datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
                        timestamp = dt.timestamp()
                    else:
                        timestamp = 0
                        
                    parsed.append({
                        'id': st_id,
                        'latitud': lat,
                        'longitud': lon,
                        'pm25': float(dp.get('valor', 0)),
                        'timestamp': timestamp
                    })
                except (ValueError, TypeError, Exception) as e:
                    continue
                
        df = pd.DataFrame(parsed)
        return self.filter_pm25(df)

    def process_wind_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        import numpy as np
        import datetime
        parsed = []
        for st in raw_data:
            lat = float(st.get('latitud', 0))
            lon = float(st.get('longitud', 0))
            st_id = st.get('codigoSerial', st.get('id', 'unknown'))
            datos_historicos = st.get('datos', [])
            
            if not datos_historicos:
                # Fallback si no tiene arreglo "datos"
                try:
                    speed = float(st.get('velocidadViento', st.get('velocidad', st.get('valor', 0.0))))
                    direction = float(st.get('direccionViento', st.get('direccion', 0.0)))
                    rad = np.radians(direction)
                    parsed.append({
                        'id': st_id,
                        'latitud': lat,
                        'longitud': lon,
                        'vx': speed * np.sin(rad),
                        'vy': speed * np.cos(rad),
                        'timestamp': st.get('fecha_hora', 0)
                    })
                except (ValueError, TypeError):
                    pass
                continue
                
            for dp in datos_historicos:
                try:
                    fecha_str = dp.get('fecha', '')
                    if fecha_str:
                        dt = datetime.datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
                        timestamp = dt.timestamp()
                    else:
                        timestamp = 0
                    
                    # Extraer velocidad y dirección de forma robusta
                    speed = float(dp.get('velocidadViento', dp.get('velocidad', dp.get('valor', 0.0))))
                    direction = float(dp.get('direccionViento', dp.get('direccion', 0.0)))
                    
                    rad = np.radians(direction)
                    parsed.append({
                        'id': st_id,
                        'latitud': lat,
                        'longitud': lon,
                        'vx': speed * np.sin(rad),
                        'vy': speed * np.cos(rad),
                        'timestamp': timestamp
                    })
                except (ValueError, TypeError, Exception):
                    continue
                    
        df = pd.DataFrame(parsed)
        # Filtrar velocidades de viento anómalas (mayores a 50 m/s)
        if not df.empty and 'vx' in df.columns:
            df = df[(df['vx'].abs() < 50.0) & (df['vy'].abs() < 50.0)]
        return df

    def run_pipeline(self):
        print("Iniciando extracción de datos SIATA (Red Oficial)...")
        
        # 1. PM2.5 Oficial
        raw_pm25 = self.fetch_data(self.endpoint_pm25)
        if raw_pm25:
            df_pm25 = self.process_pm25_data(raw_pm25)
            df_pm25.to_json("data/datos_oficiales_pm25.json", orient='records', indent=4)
            print(f"[OK] Guardados {len(df_pm25)} sensores de PM2.5 Oficial.")
        else:
            print("[WARN] No se pudo obtener la red de Calidad de Aire oficial.")
            
        # 2. Viento Oficial (Meteorología) con URLs de fallback
        endpoints_viento = [
            self.endpoint_wind,
            "https://datosabiertos.metropol.gov.co/sites/default/files/uploaded_resources/Datos_SIATA_Aire_viento.json",
            "https://datosabiertos.metropol.gov.co/sites/default/files/uploaded_resources/Datos_SIATA_Vaisala_Viento.json",
            "https://datosabiertos.metropol.gov.co/sites/default/files/uploaded_resources/Datos_SIATA_Aire_meteorologia.json"
        ]
        
        raw_wind = None
        for url in endpoints_viento:
            print(f"Intentando descargar viento desde: {url}...")
            raw_wind = self.fetch_data(url)
            if raw_wind:
                break
                
        if raw_wind:
            df_wind = self.process_wind_data(raw_wind)
            df_wind.to_json("data/datos_oficiales_viento.json", orient='records', indent=4)
            print(f"[OK] Guardados {len(df_wind)} sensores de viento Vaisala.")
        else:
            print("[WARN] No se pudo obtener la red de viento oficial en ninguno de los endpoints. Se usará el fallback de viento en Julia.")

if __name__ == "__main__":
    scraper = SiataOfficialNetwork()
    scraper.run_pipeline()
