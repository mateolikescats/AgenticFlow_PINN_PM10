import pytest
import pandas as pd
import numpy as np
from src.geo.aburra_domain import AburraDomain
from src.data.preprocessing import PINNPreprocessor

@pytest.fixture
def aburra_domain():
    return AburraDomain()

@pytest.fixture
def preprocessor(aburra_domain):
    bounds = (aburra_domain.min_lat, aburra_domain.max_lat,
              aburra_domain.min_lon, aburra_domain.max_lon)
    # bounds: min_lat, max_lat, min_lon, max_lon
    # time_max: 3600s (1 hour), conc_max: 500 ug/m3
    return PINNPreprocessor(spatial_bounds=bounds, time_max=3600.0, conc_max=500.0)

# ==================== PRUEBAS DOMINIO GEOGRÁFICO ====================
def test_aburra_domain_is_inside(aburra_domain):
    # Punto conocido dentro del valle (Medellín centro)
    assert aburra_domain.is_inside(6.2518, -75.5636) is True
    # Punto conocido fuera del valle (Bogotá)
    assert aburra_domain.is_inside(4.7110, -74.0721) is False

@pytest.mark.parametrize("lat,lon,expected", [
    (6.2518, -75.5636, True),   # Medellín centro
    (6.2000, -75.5500, True),   # Dentro del valle
    (6.4499, -75.3001, True),   # Ligeramente adentro del límite superior derecho
    (4.7110, -74.0721, False),  # Bogotá
    (8.0000, -76.0000, False),  # Fuera del área
    (6.0001, -75.6999, True),   # Ligeramente adentro de la esquina inferior izquierda
    (6.4500, -75.3000, False),  # Exactamente en el límite (Shapely contains es exclusivo en bordes)
    (6.0, -75.70, False),       # Exactamente en el límite (Shapely contains es exclusivo en bordes)
])
def test_aburra_domain_boundaries_parametrized(aburra_domain, lat, lon, expected):
    """Prueba paramétrica de múltiples puntos en/fuera del dominio"""
    assert aburra_domain.is_inside(lat, lon) is expected

def test_aburra_domain_filter_stations(aburra_domain):
    """Prueba filtrado de estaciones por dominio geográfico"""
    stations = [
        {'latitud': 6.2518, 'longitud': -75.5636, 'name': 'Medellín'},
        {'latitud': 4.7110, 'longitud': -74.0721, 'name': 'Bogotá'},
        {'latitud': 6.2000, 'longitud': -75.5500, 'name': 'Envigado'},
        {'latitud': 'invalid', 'longitud': -75.5636, 'name': 'Corrupted'},
    ]
    valid = aburra_domain.filter_stations(stations)
    assert len(valid) == 2
    assert all('name' in s for s in valid)

# ==================== PRUEBAS ESCALAMIENTO ESPACIAL ====================
def test_spatial_scaling_bounds(preprocessor, aburra_domain):
    """Verifica que los extremos del dominio mapean a [-1, 1]"""
    x_min, y_min = preprocessor.scale_spatial(aburra_domain.min_lat, aburra_domain.min_lon)
    assert pytest.approx(x_min) == -1.0
    assert pytest.approx(y_min) == -1.0

    x_max, y_max = preprocessor.scale_spatial(aburra_domain.max_lat, aburra_domain.max_lon)
    assert pytest.approx(x_max) == 1.0
    assert pytest.approx(y_max) == 1.0

def test_spatial_scaling_center(preprocessor, aburra_domain):
    """Verifica que el centro mapea a (0, 0)"""
    center_lat = (aburra_domain.min_lat + aburra_domain.max_lat) / 2
    center_lon = (aburra_domain.min_lon + aburra_domain.max_lon) / 2
    x, y = preprocessor.scale_spatial(center_lat, center_lon)
    assert pytest.approx(x, abs=1e-6) == 0.0
    assert pytest.approx(y, abs=1e-6) == 0.0

@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_spatial_scaling_monotonicity(preprocessor, aburra_domain, fraction):
    """Verifica que el escalamiento es monótonamente creciente"""
    lat = aburra_domain.min_lat + fraction * (aburra_domain.max_lat - aburra_domain.min_lat)
    lon = aburra_domain.min_lon + fraction * (aburra_domain.max_lon - aburra_domain.min_lon)
    x, y = preprocessor.scale_spatial(lat, lon)
   
    # Valor escalado debe estar en [-1, 1]
    assert -1.0 <= x <= 1.0
    assert -1.0 <= y <= 1.0
   
    # Debe ser lineal con la posición (fraction * 2 - 1)
    expected = fraction * 2.0 - 1.0
    assert pytest.approx(x, abs=1e-6) == expected
    assert pytest.approx(y, abs=1e-6) == expected

# ==================== PRUEBAS ESCALAMIENTO TEMPORAL ====================
def test_time_scaling_basic(preprocessor):
    """Verifica escalamiento básico de tiempo"""
    assert preprocessor.scale_time(0.0) == 0.0
    assert pytest.approx(preprocessor.scale_time(1800.0)) == 0.5
    assert preprocessor.scale_time(3600.0) == 1.0

def test_time_scaling_beyond_max(preprocessor):
    """Verifica comportamiento cuando el tiempo excede el máximo"""
    # Tiempo mayor al máximo debe ser mayor a 1.0
    assert preprocessor.scale_time(7200.0) == 2.0

@pytest.mark.parametrize("t_input,expected", [
    (0.0, 0.0),
    (900.0, 0.25),
    (1800.0, 0.5),
    (2700.0, 0.75),
    (3600.0, 1.0),
])
def test_time_scaling_parametrized(preprocessor, t_input, expected):
    """Prueba paramétrica del escalamiento temporal"""
    assert pytest.approx(preprocessor.scale_time(t_input)) == expected

# ==================== PRUEBAS ESCALAMIENTO CONCENTRACIÓN ====================
def test_concentration_scaling_filters_negative_noise(preprocessor):
    """Verifica filtrado de ruido (valores negativos) y escalamiento"""
    # Concentración negativa (ruido/error de sensor)
    assert preprocessor.scale_concentration(-10.5) == 0.0
    assert preprocessor.scale_concentration(-0.001) == 0.0
    # Concentración normal
    assert preprocessor.scale_concentration(250.0) == 0.5
    # Concentración extrema (saturación)
    assert preprocessor.scale_concentration(600.0) == 1.0

@pytest.mark.parametrize("conc_input,expected", [
    (0.0, 0.0),
    (125.0, 0.25),
    (250.0, 0.5),
    (375.0, 0.75),
    (500.0, 1.0),
    (750.0, 1.0),  # Saturación
    (-50.0, 0.0),  # Ruido negativo
])
def test_concentration_scaling_parametrized(preprocessor, conc_input, expected):
    """Prueba paramétrica del escalamiento de concentración"""
    assert pytest.approx(preprocessor.scale_concentration(conc_input)) == expected

def test_concentration_scaling_edge_cases(preprocessor):
    """Verifica casos límite extremos"""
    assert preprocessor.scale_concentration(0.0) == 0.0
    assert preprocessor.scale_concentration(float('inf')) == 1.0  # Saturado
    assert preprocessor.scale_concentration(float('-inf')) == 0.0  # Tratado como ruido

def test_concentration_scaling_returns_float(preprocessor):
    """Verifica que el output es siempre float"""
    result = preprocessor.scale_concentration(250)
    assert isinstance(result, float)

# ==================== PRUEBAS PROCESAMIENTO DATAFRAME ====================
def test_dataframe_processing(preprocessor):
    """Prueba completa de procesamiento de DataFrame"""
    data = {
        'latitud': [6.2518, None, 6.2000],
        'longitud': [-75.5636, -75.5000, -75.5800],
        'timestamp': [0, 1800, 3600],  # 0, 0.5, 1.0 scaled
        'pm25': [50.0, 100.0, -5.0]
    }
    df = pd.DataFrame(data)
    df_processed = preprocessor.process_dataframe(df)
   
    # Debe eliminar la fila con NaN
    assert len(df_processed) == 2
    # El valor negativo de pm25 debe ser 0
    assert df_processed.iloc[1]['u_scaled'] == 0.0
    # Timestamp escalado
    assert df_processed.iloc[0]['t_scaled'] == 0.0
    assert df_processed.iloc[1]['t_scaled'] == 1.0
   
    # Comprobar que no hay NaNs resultantes
    assert not df_processed.isnull().values.any()

def test_dataframe_processing_empty(preprocessor):
    """Verifica manejo de DataFrames vacíos"""
    df = pd.DataFrame({'latitud': [], 'longitud': [], 'timestamp': [], 'pm25': []})
    result = preprocessor.process_dataframe(df)
    assert len(result) == 0
    assert all(col in result.columns for col in ['x_scaled', 'y_scaled', 't_scaled', 'u_scaled'])

def test_dataframe_processing_all_nan(preprocessor):
    """Verifica manejo cuando todas las filas tienen NaNs"""
    data = {
        'latitud': [np.nan, np.nan],
        'longitud': [-75.5636, -75.5636],
        'timestamp': [0, 1800],
        'pm25': [50.0, 100.0]
    }
    df = pd.DataFrame(data)
    result = preprocessor.process_dataframe(df)
    assert len(result) == 0

def test_dataframe_output_columns(preprocessor):
    """Verifica que las columnas de salida son las esperadas"""
    data = {
        'latitud': [6.2518],
        'longitud': [-75.5636],
        'timestamp': [0],
        'pm25': [50.0]
    }
    df = pd.DataFrame(data)
    result = preprocessor.process_dataframe(df)
   
    expected_cols = ['latitud', 'longitud', 'timestamp', 'pm25', 'x_scaled', 'y_scaled', 't_scaled', 'u_scaled']
    for col in expected_cols:
        assert col in result.columns

def test_dataframe_output_ranges(preprocessor):
    """Verifica que todos los valores escalados están en los rangos esperados"""
    data = {
        'latitud': [6.2518, 6.2000, 6.1500],
        'longitud': [-75.5636, -75.5500, -75.4000],
        'timestamp': [0, 1800, 3600],
        'pm25': [50.0, 250.0, 600.0]
    }
    df = pd.DataFrame(data)
    result = preprocessor.process_dataframe(df)
   
    # Verificar rangos
    assert all(-1.0 <= result['x_scaled']) and all(result['x_scaled'] <= 1.0)
    assert all(-1.0 <= result['y_scaled']) and all(result['y_scaled'] <= 1.0)
    assert all(0.0 <= result['t_scaled']) and all(result['t_scaled'] <= 1.0)
    assert all(0.0 <= result['u_scaled']) and all(result['u_scaled'] <= 1.0)
