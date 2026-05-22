import pytest
import pandas as pd
from src.geo.aburra_domain import AburraDomain
from src.data.preprocessing import PINNPreprocessor

@pytest.fixture
def aburra_domain():
    return AburraDomain()

@pytest.fixture
def preprocessor(aburra_domain):
    bounds = aburra_domain.get_spatial_bounds()
    # bounds: min_lat, max_lat, min_lon, max_lon
    # time_max: 3600s (1 hour), conc_max: 500 ug/m3
    return PINNPreprocessor(spatial_bounds=bounds, time_max=3600.0, conc_max=500.0)

def test_aburra_domain_is_inside(aburra_domain):
    # Punto conocido dentro del valle (Medellín centro)
    assert aburra_domain.is_inside(6.2518, -75.5636) is True
    # Punto conocido fuera del valle (Bogotá)
    assert aburra_domain.is_inside(4.7110, -74.0721) is False

def test_spatial_scaling_bounds(preprocessor, aburra_domain):
    # Probar límites de la caja
    x_min, y_min = preprocessor.scale_spatial(aburra_domain.min_lat, aburra_domain.min_lon)
    assert pytest.approx(x_min) == -1.0
    assert pytest.approx(y_min) == -1.0

    x_max, y_max = preprocessor.scale_spatial(aburra_domain.max_lat, aburra_domain.max_lon)
    assert pytest.approx(x_max) == 1.0
    assert pytest.approx(y_max) == 1.0

def test_concentration_scaling_filters_negative_noise(preprocessor):
    # Concentración negativa (ruido/error de sensor)
    assert preprocessor.scale_concentration(-10.5) == 0.0
    # Concentración normal
    assert preprocessor.scale_concentration(250.0) == 0.5
    # Concentración extrema (saturación)
    assert preprocessor.scale_concentration(600.0) == 1.0

def test_dataframe_processing(preprocessor):
    data = {
        'latitud': [6.2518, None, 6.2000],
        'longitud': [-75.5636, -75.5000, -75.5800],
        'timestamp': [0, 1800, 3600], # 0, 0.5, 1.0 scaled
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
