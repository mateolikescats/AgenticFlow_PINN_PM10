import folium
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DomainMapGenerator:
    """
    Generador de mapa interactivo para validación visual de la Fase 1.
    Permite verificar que los sensores caigan estrictamente dentro
    del bounding box geográfico del Valle de Aburrá.
    """
    
    def __init__(self, bounds: tuple):
        """
        :param bounds: (min_lat, max_lat, min_lon, max_lon)
        """
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = bounds
        # Centro aproximado de Medellín
        self.center_lat = (self.min_lat + self.max_lat) / 2
        self.center_lon = (self.min_lon + self.max_lon) / 2
        self.map = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=11)
        
    def _add_domain_boundaries(self):
        """
        Dibuja el polígono/bounding box que representa el dominio matemático de la PINN.
        """
        points = [
            (self.min_lat, self.min_lon),
            (self.min_lat, self.max_lon),
            (self.max_lat, self.max_lon),
            (self.max_lat, self.min_lon),
            (self.min_lat, self.min_lon)
        ]
        folium.Polygon(
            locations=points,
            color='red',
            weight=2,
            fill=True,
            fill_opacity=0.1,
            tooltip="Dominio Espacial PINN (Valle de Aburrá)"
        ).add_to(self.map)
        
    def plot_stations(self, stations: List[Dict]):
        """
        Grafica los sensores en el mapa.
        Espera una lista de diccionarios con 'latitud', 'longitud', 'id' y 'pm25'.
        """
        self._add_domain_boundaries()
        
        for st in stations:
            lat = st.get('latitud')
            lon = st.get('longitud')
            pm25 = st.get('pm25', 0)
            st_id = st.get('id', 'N/A')
            
            if lat is not None and lon is not None:
                color = 'green' if pm25 < 35 else 'orange' if pm25 < 50 else 'red'
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=f"Sensor ID: {st_id}<br>PM2.5: {pm25} µg/m³"
                ).add_to(self.map)

    def save_map(self, output_path: str = "mapa_validacion.html"):
        """Guarda el mapa interactivo en un archivo HTML."""
        self.map.save(output_path)
        logger.info(f"Mapa interactivo guardado en: {output_path}")

if __name__ == "__main__":
    from src.geo.aburra_domain import AburraDomain
    logging.basicConfig(level=logging.INFO)
    # Prueba con datos mockeados para validación visual
    domain = AburraDomain()
    mapper = DomainMapGenerator(domain.get_spatial_bounds())
    
    mock_stations = [
        {'id': 'C-001', 'latitud': 6.2518, 'longitud': -75.5636, 'pm25': 25}, # Centro (Verde)
        {'id': 'C-002', 'latitud': 6.1500, 'longitud': -75.6000, 'pm25': 45}, # Sur (Naranja)
        {'id': 'C-003', 'latitud': 6.3000, 'longitud': -75.5000, 'pm25': 60}, # Norte (Rojo)
        {'id': 'C-004', 'latitud': 5.0000, 'longitud': -74.0000, 'pm25': 10}, # Fuera (Debe ignorarse si se filtra antes)
    ]
    
    valid_stations = domain.filter_stations(mock_stations)
    mapper.plot_stations(valid_stations)
    mapper.save_map("mapa_validacion.html")
    print("Se ha generado 'mapa_validacion.html' con éxito.")
