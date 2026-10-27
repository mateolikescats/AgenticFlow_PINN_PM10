import requests
import pandas as pd
import logging
import json
from typing import List, Dict
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class SiataOfficialNetwork:
    """
    Cliente para la ingesta de datos de la Red Oficial de Calidad de Aire y Meteorología del SIATA.
    Se separa la calidad de aire del viento, dado que muy pocos sensores miden ambas cosas al tiempo.
    """

    def __init__(self):
        # El portal datosabiertos.metropol.gov.co dejó de responder (502) y los JSON estáticos de
        # siata.gov.co/EntregaData1 están congelados (PM2.5 hasta 2024-07, viento hasta 2022).
        # Se usa la API del geoportal de SIATA, que entrega las últimas 72 h de PM2.5 por estación
        # y la última lectura de viento de la red meteorológica.
        self.base_url = "https://geoportal.siata.gov.co/fastgeoapi/geodata"
        self.endpoint_pm25_stations = [
            f"{self.base_url}/geodataJson/1/pm25_minio",
            f"{self.base_url}/geodataJson/1/pm25No_Poeca",
        ]
        self.endpoint_pm25_series = f"{self.base_url}/geographJson/1/pm25/"
        self.endpoint_wind = f"{self.base_url}/geodataJson/3/tempVient"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SIATA-Research-Agent/2.0',
            'Accept': 'application/json'
        })
        # Deshabilitar advertencias de SSL rotos típicos de dominios gubernamentales
        import urllib3
        urllib3.disable_warnings()

    def fetch_data(self, url: str):
        """Descarga JSON en tiempo real desde un endpoint específico."""
        try:
            # verify=False porque SIATA frecuentemente tiene problemas con el certificado SSL
            response = self.session.get(url, timeout=30, verify=False)
            response.raise_for_status()
            data = response.json()
            if not data:
                return None
            return data
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Error HTTP al contactar {url}: {e}")
            return None

    def filter_pm25(self, df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
        """Filtro estricto para red oficial (aislando anomalías extremas)."""
        if df.empty or 'pm25' not in df.columns:
            return df
            
        df_clean = df[(df['pm25'] >= 0) & (df['pm25'] < 500)].copy()
        
        if len(df_clean) < 5:
            return df_clean
            
        # El IsolationForest se aplica sobre la desviación respecto a la mediana móvil de cada
        # estación (picos aislados), no sobre el nivel absoluto: sobre el nivel eliminaba
        # sistemáticamente los episodios reales de alta concentración.
        df_clean = df_clean.sort_values(['id', 'timestamp'])
        mediana_local = df_clean.groupby('id')['pm25'].transform(
            lambda s: s.rolling(5, center=True, min_periods=1).median())
        X = (df_clean['pm25'] - mediana_local).abs().to_numpy().reshape(-1, 1)
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(X)
        return df_clean[preds == 1]

    def process_pm25_data(self, stations: List[Dict]) -> pd.DataFrame:
        """Descarga la serie horaria de 72 h de cada estación y la convierte a registros (id, lat, lon, pm25, timestamp)."""
        import datetime
        parsed = []
        for st in stations:
            props = st['properties']
            lon, lat = st['geometry']['coordinates'][:2]
            st_id = int(props['codigo'])
            serie = self.fetch_data(f"{self.endpoint_pm25_series}{st_id}")
            if not serie or 'info' not in serie:
                logger.warning(f"Sin serie de PM2.5 para la estación {st_id}")
                continue
            info = serie['info']
            valores = info.get('PM25_72H', [])
            hora_fin = datetime.datetime.strptime(info['Hora_fin'], "%Y-%m-%d %H:%M:%S")
            n = len(valores)
            municipio = props.get('Municipio', '').strip()
            nombre = props.get('nombreEstacion', str(st_id))
            if municipio and not nombre.lower().startswith(municipio.lower()):
                nombre = f"{municipio} - {nombre}"
            for i, valor in enumerate(valores):
                if valor is None:
                    continue
                dt = hora_fin - datetime.timedelta(hours=n - 1 - i)
                parsed.append({
                    'id': st_id,
                    'nombre': nombre,
                    'latitud': float(lat),
                    'longitud': float(lon),
                    'pm25': float(valor),
                    'fecha': dt.strftime("%Y-%m-%d %H:%M:%S"),
                    'timestamp': dt.timestamp()
                })
        df = pd.DataFrame(parsed)
        return self.filter_pm25(df)

    def process_wind_data(self, raw_data: Dict) -> pd.DataFrame:
        import numpy as np
        import datetime
        parsed = []
        for feat in raw_data.get('features', []):
            props = feat['properties']
            lon, lat = feat['geometry']['coordinates'][:2]
            try:
                speed = float(props['Magnitud_viento'])
                direction = float(props['Direccion_viento'])
                dt = datetime.datetime.strptime(props['FechaUltimaModificacion'], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError, TypeError):
                continue
            # La dirección meteorológica indica de DÓNDE viene el viento; el vector físico
            # de velocidad apunta hacia donde va, por eso el signo negativo.
            rad = np.radians(direction)
            parsed.append({
                'id': int(props['Codigo']),
                'latitud': float(lat),
                'longitud': float(lon),
                'vx': -speed * np.sin(rad),
                'vy': -speed * np.cos(rad),
                'timestamp': dt.timestamp()
            })
                    
        df = pd.DataFrame(parsed)
        # Filtrar velocidades de viento anómalas (mayores a 50 m/s)
        if not df.empty and 'vx' in df.columns:
            df = df[(df['vx'].abs() < 50.0) & (df['vy'].abs() < 50.0)]
        return df

    def run_pipeline(self):
        print("Iniciando extracción de datos SIATA (Geoportal, últimas 72 h)...")
        
        # 1. PM2.5 Oficial (estaciones POECA y no-POECA, sin duplicados)
        stations = {}
        for url in self.endpoint_pm25_stations:
            data = self.fetch_data(url)
            for feat in (data or {}).get('features', []):
                stations[feat['properties']['codigo']] = feat
        if stations:
            df_pm25 = self.process_pm25_data(list(stations.values()))
            df_pm25.to_json("data/datos_oficiales_pm25.json", orient='records', indent=4, force_ascii=False)
            print(f"[OK] Guardados {len(df_pm25)} registros de PM2.5 de {df_pm25['id'].nunique()} estaciones "
                  f"({df_pm25['fecha'].min()} -> {df_pm25['fecha'].max()}).")
        else:
            print("[WARN] No se pudo obtener la red de Calidad de Aire oficial.")
            
        # 2. Viento Oficial (Meteorología)
        raw_wind = self.fetch_data(self.endpoint_wind)
        if raw_wind:
            df_wind = self.process_wind_data(raw_wind)
            df_wind.to_json("data/datos_oficiales_viento.json", orient='records', indent=4)
            print(f"[OK] Guardados {len(df_wind)} registros de viento.")
        else:
            print("[WARN] No se pudo obtener la red de viento oficial. Se usará el viento simulado del preprocesador.")

if __name__ == "__main__":
    scraper = SiataOfficialNetwork()
    scraper.run_pipeline()
