import requests
import pandas as pd
import logging
from typing import Optional, List, Dict
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class SiataCitizenNetwork:
    """
    Cliente para la ingesta de datos de la Red de Ciudadanos Científicos del SIATA.
    Se encarga de la comunicación HTTP, parseo y un filtrado primario de anomalías
    usando Isolation Forest para descartar sensores descalibrados.
    """

    #? Un script de ingesta dedicado a SIATA es crucial para asegurar que obtenemos datos limpios y confiables.
    
    def __init__(self, endpoint_url: str = "https://siata.gov.co/ciudadano_cientifico/php/estaciones/view_estaciones_json.php"):

        #? Se configura una sesión HTTP con headers personalizados para evitar bloqueos por parte de SIATA, que podría detectar tráfico automatizado.
        #? Al poner Mozilla/5.0, simulamos un navegador común, y el User-Agent específico ayuda a identificar que somos un agente de investigación legítimo.
        #? Así, el equipo de SIATA pueda contactarnos si hay problemas.

        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SIATA-Research-Agent/1.0',
            'Accept': 'application/json'
        })

    def fetch_live_data(self) -> Optional[List[Dict]]:

        #? Esto hace la petición real a la API de SIATA. 
        #? Tiene un timeout de 15 segundos; si el servidor no responde, se maneja la excepción y se retorna None para que la capa superior decida qué hacer (ej. usar datos mockeados o reintentar más tarde).

        """
        Descarga el JSON en tiempo real de la API.
        En un entorno real, manejaría autenticación o tokens si la API no es 100% abierta.
        """
        try:
            # Nota: Este request puede fallar en producción si SIATA bloquea IPs de centros de datos.
            response = self.session.get(self.endpoint_url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                logger.error("JSON vacío recibido de SIATA.")
                return None
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error HTTP al contactar SIATA: {e}")
            # Retorna None para que la capa superior decida si usar datos mockeados o reintentar
            return None

    def filter_anomalies(self, df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:

        #? Funciona en dos etapas: primero un filtro básico para eliminar valores obviamente erróneos (negativos o extremadamente altos; como un filtro físico).
        #? Luego un filtro más sofisticado con Isolation Forest para detectar patrones anómalos que podrían indicar sensores descalibrados o fallas temporales.
        #? Una concentración negativa de PM2.5 no tiene sentido físico, y valores por encima de 1000 ug/m3 son extremadamente improbables en condiciones normales (puede indicar un sensor roto o un error de lectura), por lo que se eliminan de inmediato.
        """
        Aplica Isolation Forest para identificar y descartar picos anómalos 
        (ej. un sensor reportando 999 ug/m3 de pm2.5 súbitamente por fallo de hardware).
        
        Args:
            df: DataFrame con columna 'pm25'
            contamination: Porcentaje estimado de datos anómalos.
        """
        if df.empty or 'pm25' not in df.columns:
            return df
            
        # Filtro básico: Valores negativos o absurdamente altos (>1000)
        df_clean = df[(df['pm25'] >= 0) & (df['pm25'] < 1000)].copy() #? Se asegura que las leyes de la física se respeten, eliminando valores imposibles que podrían distorsionar el modelo.
        
        if len(df_clean) < 10:
            logger.warning("Muy pocos datos para ejecutar Isolation Forest. Retornando filtro básico.")
            return df_clean
            
        clf = IsolationForest(contamination=contamination, random_state=42) 
        #? El filtro estadístico de Isolation Forest es útil para detectar patrones anómalos que no son evidentes con un simple filtro de rango.
        #? El parámetro de contaminación se ajusta para reflejar la expectativa de que alrededor del 5% de los datos podrían ser anómalos, lo cual es razonable en un entorno de sensores ciudadanos donde pueden ocurrir fallas ocasionales.
        # Reshape para scikit-learn
        X = df_clean['pm25'].to_numpy().reshape(-1, 1)
        preds = clf.fit_predict(X) #? Este método devuelve 1 para inliers (datos normales) y -1 para outliers (datos anómalos). Al filtrar solo los inliers, se mejora la calidad de los datos que se usarán para entrenar el modelo de predicción, reduciendo el ruido causado por sensores defectuosos o lecturas erróneas.
        
        # IsolationForest retorna 1 para inliers y -1 para outliers
        df_clean = df_clean[preds == 1]
        
        logger.info(f"Isolation Forest descartó {len(df) - len(df_clean)} registros anómalos.")
        return df_clean

    def get_clean_dataframe(self, raw_data: List[Dict]) -> pd.DataFrame:

        #? Esta función se encarga de convertir el JSON crudo de SIATA en un DataFrame estructurado, asegurándose de manejar cualquier inconsistencia en los datos (como campos faltantes o tipos incorrectos) 
        #? Luego aplicar el filtro de anomalías para obtener un conjunto de datos limpio y listo para análisis o modelado.
        """
        Convierte el JSON de SIATA en un DataFrame estructurado y limpio.
        """
        # Parseo seguro (las llaves exactas dependerán del contrato final de la API)
        parsed = []
        for st in raw_data:
            try:
                parsed.append({
                    'id': st.get('codigo', st.get('id', 'unknown')),
                    'latitud': float(st.get('latitud', 0)),
                    'longitud': float(st.get('longitud', 0)),
                    'pm25': float(st.get('pm25', 0)),
                    'timestamp': st.get('fecha_hora', 0) # En la práctica se debe parsear a datetime
                })
            except (ValueError, TypeError): #? Si el JSON tiene valores no numéricos en campos que deberían ser numéricos, se captura la excepción y se omite ese registro, evitando que datos corruptos contaminen el DataFrame final.
                continue
                
        df = pd.DataFrame(parsed)
        return self.filter_anomalies(df)
