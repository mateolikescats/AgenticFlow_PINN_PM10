import logging
from shapely.geometry import Point, box

logger = logging.getLogger(__name__)

class AburraDomain:
    """
    Define el dominio geográfico Omega para la ecuación ADR de la PINN.
    El Valle de Aburrá está geográficamente restringido. Esta clase asegura
    que las coordenadas utilizadas para el entrenamiento pertenezcan estrictamente
    al dominio físico, mitigando el problema de 'ill-posedness' por datos fuera
    de la topología (\\Omega).
    """

    #? Básicamente con esto se restringe el dominio geográfico a un área específica, evitando que el modelo aprenda patrones de datos que no corresponden al área de interés. 
    #? Esto es crucial para garantizar que el modelo se enfoque en aprender la dinámica de la contaminación del aire dentro del Valle de Aburrá, y no en áreas circundantes que podrían tener características muy diferentes. 
    
    def __init__(self):
        # Bounding box aproximado del Valle de Aburrá (Caldas a Barbosa)
        # lon_min, lat_min, lon_max, lat_max
        self.min_lon = -75.70
        self.min_lat = 6.00
        self.max_lon = -75.30
        self.max_lat = 6.45
        
        # Representación geométrica en R^2
        self.bbox = box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)
        #? La función box de Shapely crea un polígono rectangular definido por las coordenadas mínimas y máximas de longitud y latitud. 
        #? Esto permite realizar operaciones geométricas como verificar si un punto está dentro del área definida por el bounding box.

    def is_inside(self, lat: float, lon: float) -> bool:
        """
        Verifica si un punto está dentro del bounding box del Valle de Aburrá.
        """
        point = Point(lon, lat)
        return self.bbox.contains(point)

    def filter_stations(self, stations_data: list) -> list:
        """
        Filtra una lista de estaciones, conservando solo aquellas dentro del Valle.
        Esperado que cada estación tenga 'latitud' y 'longitud'.
        """
        valid_stations = []
        for st in stations_data:
            try:
                lat = float(st.get('latitud', 0))
                lon = float(st.get('longitud', 0))
                if self.is_inside(lat, lon):
                    valid_stations.append(st)
                else:
                    logger.warning(f"Estación excluida (fuera del dominio \\Omega): lat={lat}, lon={lon}")
            except (ValueError, TypeError):
                continue
        return valid_stations

    def get_spatial_bounds(self):
        """
        Retorna los límites espaciales (min_lat, max_lat, min_lon, max_lon)
        necesarios para la adimensionalización.
        """
        return self.min_lat, self.max_lat, self.min_lon, self.max_lon
