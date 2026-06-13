import requests

def get_elevations(lats, lons):
    url = "https://elevation-api.open-meteo.com/v1/elevation"
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons))
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("elevation", [])
    else:
        print(f"Error fetching elevation: {response.status_code}")
        return []

# Test coordinates from screenshot (Estación 28 and 88)
lats = [6.1856666, 6.1686831]
lons = [-75.5972061, -75.5819702]

elevations = get_elevations(lats, lons)
print(f"Elevations: {elevations}")
