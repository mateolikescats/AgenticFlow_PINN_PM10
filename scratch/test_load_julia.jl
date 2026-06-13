using JSON

println("=== JULIA DIRECT DATA LOADING AND LOOKUP ===")

# 1. Load wind data and build lookup table
wind_path = "datos_meteorologicos_viento.json"
if !isfile(wind_path)
    println("Error: wind file not found.")
    exit(1)
end

wind_data = JSON.parsefile(wind_path)
station_coords = Dict{Int, Dict{String, Float64}}()

function clean_wind_id(id_str::String)
    clean_str = replace(id_str, "W-" => "")
    return round(Int, parse(Float64, clean_str))
end

for d in wind_data
    id_str = d["id"]
    id_clean = clean_wind_id(id_str)
    if !haskey(station_coords, id_clean)
        station_coords[id_clean] = Dict(
            "x" => Float64(d["x"]),
            "y" => Float64(d["y"]),
            "z" => Float64(d["z"]),
            "elevacion_real" => Float64(d["elevacion_real"]),
            "latitud" => Float64(d["latitud"]),
            "longitud" => Float64(d["longitud"])
        )
    end
end

println("Loaded station coordinates for ", length(station_coords), " stations.")
println("Sample station 3 coords: ", get(station_coords, 3, "Not found"))

# 2. Load raw PM2.5 data
pm_path = "datos_oficiales_pm25.json"
if !isfile(pm_path)
    println("Error: PM2.5 file not found.")
    exit(1)
end

pm_raw = JSON.parsefile(pm_path)
println("Loaded raw PM2.5 records: ", length(pm_raw))

# 3. Clean and sort PM2.5 data
# Filter out NaNs or missing keys, and duplicate records by station and timestamp
seen_records = Set{Tuple{Int, Float64}}()
cleaned_pm = []
for d in pm_raw
    id = round(Int, d["id"])
    t_val = Float64(d["timestamp"])
    pm_val = Float64(d["pm25"])
    
    # Check if duplicate
    rec_key = (id, t_val)
    if rec_key in seen_records
        continue
    end
    
    # Check if station exists in coords lookup
    if !haskey(station_coords, id)
        continue
    end
    
    push!(seen_records, rec_key)
    push!(cleaned_pm, Dict("id" => id, "timestamp" => t_val, "pm25" => pm_val))
end

println("Cleaned records (removed duplicates and unmatched stations): ", length(cleaned_pm))

# Sort cronologically by timestamp and then by id
sort!(cleaned_pm, by = d -> (d["timestamp"], d["id"]))

# 4. Scale PM2.5 and temperature
timestamps = [d["timestamp"] for d in cleaned_pm]
t_min = minimum(timestamps)
t_max = maximum(timestamps)
t_range = t_max - t_min

final_pm_data = []
for d in cleaned_pm
    id = d["id"]
    coords = station_coords[id]
    
    t_scaled = t_range > 0 ? (d["timestamp"] - t_min) / t_range : 0.0
    u_scaled = clamp(d["pm25"] / 100.0, 0.0, 1.0)
    T_scaled = 2.0 * coords["z"] - 1.0
    
    push!(final_pm_data, Dict(
        "id" => string(id),
        "x" => coords["x"],
        "y" => coords["y"],
        "z" => coords["z"],
        "t" => t_scaled,
        "u" => u_scaled,
        "T" => T_scaled,
        "elevacion_real" => coords["elevacion_real"],
        "pm25" => d["pm25"],
        "latitud" => coords["latitud"],
        "longitud" => coords["longitud"]
    ))
end

println("Successfully processed ", length(final_pm_data), " PM2.5 records.")
println("Sample processed record 1: ", final_pm_data[1])
