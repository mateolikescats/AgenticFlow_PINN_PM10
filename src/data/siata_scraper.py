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

    def run_pipeline(self):
        print("Iniciando extracción de datos SIATA (Red Oficial)...")
        
        # PM2.5 Oficial
        raw_pm25 = self.fetch_data(self.endpoint_pm25)
        if raw_pm25:
            df_pm25 = self.process_pm25_data(raw_pm25)
            df_pm25.to_json("datos_oficiales_pm25.json", orient='records', indent=4)
            print(f"[OK] Guardados {len(df_pm25)} sensores de PM2.5 Oficial.")
        else:
            print("[WARN] No se pudo obtener la red de Calidad de Aire oficial.")

if __name__ == "__main__":
    scraper = SiataOfficialNetwork()
    scraper.run_pipeline()
