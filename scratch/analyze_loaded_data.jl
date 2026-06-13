using Pkg
Pkg.activate(".")
using JSON, Statistics

function clean_wind_id(id_str::String)
    clean_str = replace(id_str, "W-" => "")
    return round(Int, parse(Float64, clean_str))
end

function load_data(filepath)
    return JSON.parsefile(filepath)
end

function load_processed_data(pm_path::String, wind_path::String)
    wind_data = load_data(wind_path)
    station_coords = Dict{Int, Dict{String, Float64}}()

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

    pm_raw = load_data(pm_path)
    seen_records = Set{Tuple{Int, Float64}}()
    cleaned_pm = []

    for d in pm_raw
        id = round(Int, d["id"])
        t_val = Float64(d["timestamp"])
        pm_val = Float64(d["pm25"])
        
        rec_key = (id, t_val)
        if rec_key in seen_records
            continue
        end
        if !haskey(station_coords, id)
            continue
        end
        
        push!(seen_records, rec_key)
        push!(cleaned_pm, Dict("id" => id, "timestamp" => t_val, "pm25" => pm_val))
    end

    sort!(cleaned_pm, by = d -> (d["timestamp"], d["id"]))

    timestamps = [d["timestamp"] for d in cleaned_pm]
    t_min = minimum(timestamps)
    t_max = maximum(timestamps)
    t_range = t_max - t_min

    final_pm_data = Dict{String, Any}[]
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

    return final_pm_data
end

function analyze()
    data = load_processed_data("datos_oficiales_pm25.json", "datos_meteorologicos_viento.json")
    
    pm25_vals = [d["pm25"] for d in data]
    u_vals = [d["u"] for d in data]
    t_vals = [d["t"] for d in data]
    x_vals = [d["x"] for d in data]
    y_vals = [d["y"] for d in data]
    z_vals = [d["z"] for d in data]
    
    println("Aligned Data Statistics:")
    println("  - Total Aligned Records: ", length(data))
    println("  - PM2.5: range = [", minimum(pm25_vals), ", ", maximum(pm25_vals), "], mean = ", mean(pm25_vals), ", std = ", std(pm25_vals))
    println("  - u (scaled): range = [", minimum(u_vals), ", ", maximum(u_vals), "], mean = ", mean(u_vals), ", std = ", std(u_vals))
    println("  - t (scaled): range = [", minimum(t_vals), ", ", maximum(t_vals), "], mean = ", mean(t_vals))
    println("  - x: range = [", minimum(x_vals), ", ", maximum(x_vals), "]")
    println("  - y: range = [", minimum(y_vals), ", ", maximum(y_vals), "]")
    println("  - z: range = [", minimum(z_vals), ", ", maximum(z_vals), "]")
end

analyze()
