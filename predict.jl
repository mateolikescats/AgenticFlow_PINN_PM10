using Pkg
Pkg.activate(".")
include("src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE, ModelingToolkit, JLD2, ComponentArrays, Lux, JSON

struct GridPoint
    lon::Float64
    lat::Float64
    elev::Float64
    S::Float64
    vx::Float64
    vy::Float64
    x::Float64
    y::Float64
    z::Float64
end

function run_prediction(input_path::String="input_points.json", output_path::String="output_predictions.json", model_path::String="modelo_pinn.jld2")
    println("==== Predictor Standalone de la iPINN ====")
    
    # 1. Verificar existencia del modelo entrenado
    if !isfile(model_path)
        println("❌ Error: No se encontró el archivo del modelo en $model_path. Por favor entrena la PINN primero.")
        return
    end

    # 2. Generar archivo de entrada de ejemplo si no existe
    if !isfile(input_path)
        println("📝 Creando archivo de ejemplo '$input_path' con coordenadas reales de Medellín...")
        example_data = [
            Dict("latitud" => 6.2518, "longitud" => -75.5636, "elevacion" => 1500.0, "timestamp" => 0.0), # Medellín Centro (t=0)
            Dict("latitud" => 6.2000, "longitud" => -75.5500, "elevacion" => 1600.0, "timestamp" => 0.5), # Envigado (t=0.5)
            Dict("latitud" => 6.3000, "longitud" => -75.5800, "elevacion" => 1800.0, "timestamp" => 1.0)  # Bello (Ladera) (t=1.0)
        ]
        open(input_path, "w") do f
            JSON.print(f, example_data, 4)
        end
    end

    # 3. Cargar coordenadas de entrada
    println("Cargando coordenadas de entrada desde '$input_path'...")
    inputs = JSON.parsefile(input_path)
    N = length(inputs)
    println("Procesando $N puntos de entrada...")

    # 4. Adimensionalizar coordenadas (Idéntico al Preprocesamiento)
    # Bounds del Valle de Aburrá
    lat_min, lat_max = 6.0, 6.45
    lon_min, lon_max = -75.7, -75.3
    elev_min, elev_max = 1400.0, 3000.0

    pts = Matrix{Float64}(undef, 4, N)
    for i in 1:N
        d = inputs[i]
        lat = Float64(d["latitud"])
        lon = Float64(d["longitud"])
        elev = Float64(d["elevacion"])
        t_val = Float64(d["timestamp"]) # Se asume escalado en [0, 1]

        # Escalar x, y a [-1, 1]
        x_scaled = 2.0 * (lon - lon_min) / (lon_max - lon_min) - 1.0
        y_scaled = 2.0 * (lat - lat_min) / (lat_max - lat_min) - 1.0
        # Escalar z a [0, 1]
        z_scaled = (elev - elev_min) / (elev_max - elev_min)

        pts[1, i] = x_scaled
        pts[2, i] = y_scaled
        pts[3, i] = z_scaled
        pts[4, i] = t_val
    end

    # 5. Recrear arquitectura PINN y cargar pesos
    println("Inicializando redes neuronales...")
    pdesys, _ = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())
    adaptive_strategy = GradientScaleAdaptiveLoss(100)
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=(phi, θ, p)->0.0, weight_strategy=adaptive_strategy)
    
    println("Cargando pesos entrenados desde '$model_path'...")
    @load model_path theta
    phi = discretization.phi

    # 6. Evaluar predicciones de las redes correspondientes
    println("Evaluando red neuronal de la PINN Inversa...")
    u_pred = phi[1](pts, theta.depvar.u) # Concentración PM2.5
    T_pred = phi[2](pts, theta.depvar.T) # Temperatura
    vx_pred = phi[3](pts, theta.depvar.vx) # Viento X
    vy_pred = phi[4](pts, theta.depvar.vy) # Viento Y
    vz_pred = phi[5](pts, theta.depvar.vz) # Viento Z
    S_pred = phi[7](pts, theta.depvar.S) # PINN Inversa: Fuentes de Emisión

    # 7. Calcular Trayectorias No-Lineales usando la Red Neuronal de la PINN directamente
    println("Calculando trayectorias físicas no-lineales a partir de la red neuronal...")
    trajectories_list = Vector{Vector{Vector{Float64}}}(undef, N)
    dt_scaled = 0.055  # Paso temporal escalado
    for i in 1:N
        x = pts[1, i]
        y = pts[2, i]
        z = pts[3, i]
        t = pts[4, i]
        
        path = Vector{Float64}[]
        # Registrar posición inicial (real)
        lon_init = lon_min + (x + 1.0)/2.0 * (lon_max - lon_min)
        lat_init = lat_min + (y + 1.0)/2.0 * (lat_max - lat_min)
        elev_init = elev_min + z * (elev_max - elev_min)
        push!(path, [lon_init, lat_init, elev_init])
        
        # Simular 15 pasos de advección física
        for step in 1:15
            pt = reshape([x, y, z, t], 4, 1)
            # Evaluar vientos directamente de la red neuronal en esta coordenada exacta
            vx = phi[3](pt, theta.depvar.vx)[1]
            vy = phi[4](pt, theta.depvar.vy)[1]
            vz = phi[5](pt, theta.depvar.vz)[1]
            
            # Actualizar posición en espacio adimensional
            x = clamp(x + vx * dt_scaled, -1.0, 1.0)
            y = clamp(y + vy * dt_scaled, -1.0, 1.0)
            z = clamp(z + vz * dt_scaled, 0.0, 1.0)
            
            # Convertir a coordenadas reales para visualización
            lon = lon_min + (x + 1.0)/2.0 * (lon_max - lon_min)
            lat = lat_min + (y + 1.0)/2.0 * (lat_max - lat_min)
            elev = elev_min + z * (elev_max - elev_min)
            push!(path, [lon, lat, elev])
        end
        trajectories_list[i] = path
    end

    # 8. Des-adimensionalizar y consolidar resultados
    println("Consolidando resultados y guardando en '$output_path'...")
    output_data = Dict{String, Any}[]
    for i in 1:N
        d = inputs[i]
        
        # Des-escalar variables físicas (pm25_max = 100, viento_max = 10)
        pm25_est = clamp(u_pred[i] * 100.0, 0.0, Inf)
        vx_est = vx_pred[i] * 10.0
        vy_est = vy_pred[i] * 10.0
        vz_est = vz_pred[i] * 10.0
        
        # El término de emisión S (fuente predicha de PM2.5 por la PINN Inversa)
        # S_pred está en unidades adimensionales por segundo. Lo des-escalamos a ug/(m3 * s)
        emision_est = S_pred[i] * (100.0 / 3600.0) # Concentracion_max / Tiempo_max

        push!(output_data, Dict(
            "latitud" => d["latitud"],
            "longitud" => d["longitud"],
            "elevacion" => d["elevacion"],
            "timestamp" => d["timestamp"],
            "pred_pm25_ug_m3" => round(pm25_est, digits=3),
            "pred_viento_vx_m_s" => round(vx_est, digits=3),
            "pred_viento_vy_m_s" => round(vy_est, digits=3),
            "pred_viento_vz_m_s" => round(vz_est, digits=3),
            "pred_emision_S_ug_m3_s" => round(emision_est, digits=6), # <-- Predicción de la PINN Inversa!
            "trajectory" => trajectories_list[i] # <-- Trayectoria física calculada directamente por la PINN!
        ))
    end

    open(output_path, "w") do f
        JSON.print(f, output_data, 4)
    end
    println("✅ ¡Predicciones guardadas exitosamente en '$output_path'!")

    # ==========================================================================
    # NUEVO: BÚSQUEDA EN GRILLA DE FUENTES DE EMISIÓN REALES (LOCALIZACIÓN INVERSA)
    # ==========================================================================
    println("\n🔎 Buscando focos de emisión (fuentes) en todo el valle...")
    grid_lon = range(lon_min + 0.03, lon_max - 0.03, length=35)
    grid_lat = range(lat_min + 0.03, lat_max - 0.03, length=35)
    N_grid = length(grid_lon) * length(grid_lat)
    
    grid_pts = Matrix{Float64}(undef, 4, N_grid)
    idx = 1
    for lon in grid_lon, lat in grid_lat
        x_scaled = 2.0 * (lon - lon_min) / (lon_max - lon_min) - 1.0
        y_scaled = 2.0 * (lat - lat_min) / (lat_max - lat_min) - 1.0
        
        # Estimar elevación de la grilla interpolando las estaciones sensoras
        # en lugar de usar un z_scaled constante de 0.05 (que queda bajo tierra en las montañas)
        w_sum = 0.0
        elev_sum = 0.0
        for st in inputs
            dist = sqrt((lon - Float64(st["longitud"]))^2 + (lat - Float64(st["latitud"]))^2)
            if dist < 1e-5
                elev_sum = Float64(st["elevacion"])
                w_sum = 1.0
                break
            end
            w = 1.0 / (dist^2)
            w_sum += w
            elev_sum += Float64(st["elevacion"]) * w
        end
        elev_val = elev_sum / w_sum
        
        z_scaled = clamp((elev_val - elev_min) / (elev_max - elev_min), 0.0, 1.0)
        t_scaled = 1.0  # Último timestamp
        
        grid_pts[1, idx] = x_scaled
        grid_pts[2, idx] = y_scaled
        grid_pts[3, idx] = z_scaled
        grid_pts[4, idx] = t_scaled
        idx += 1
    end
    
    # Evaluar red de emisión S en la grilla
    S_grid = phi[7](grid_pts, theta.depvar.S)
    emision_grid = S_grid * (100.0 / 3600.0) # des-escalar a ug/m3/s
    
    # Evaluar vientos de la grilla para tener los vectores en los focos
    vx_grid = phi[3](grid_pts, theta.depvar.vx) * 10.0
    vy_grid = phi[4](grid_pts, theta.depvar.vy) * 10.0
    
    # Estructurar puntos
    
    all_points = GridPoint[]
    idx = 1
    for lon in grid_lon, lat in grid_lat
        # Calcular elevación interpolada para este punto
        w_sum = 0.0
        elev_sum = 0.0
        for st in inputs
            dist = sqrt((lon - Float64(st["longitud"]))^2 + (lat - Float64(st["latitud"]))^2)
            if dist < 1e-5
                elev_sum = Float64(st["elevacion"])
                w_sum = 1.0
                break
            end
            w = 1.0 / (dist^2)
            w_sum += w
            elev_sum += Float64(st["elevacion"]) * w
        end
        elev_val = elev_sum / w_sum

        # Filtrar puntos para mantenerlos estrictamente cerca del fondo del valle (cerca de estaciones urbanas)
        # Esto previene que se ubiquen focos en las cumbres o laderas altas de las montañas
        is_near_urban_station = false
        for st in inputs
            # Excluir estaciones de alta montaña (> 1800 msnm, como Santa Elena) de definir el valle urbano
            if Float64(st["elevacion"]) >= 1800.0
                continue
            end
            dist = sqrt((lon - Float64(st["longitud"]))^2 + (lat - Float64(st["latitud"]))^2)
            if dist <= 0.022  # Radio estrecho de ~2.4 km para confinarlos al plano urbano del valle
                is_near_urban_station = true
                break
            end
        end
        
        if is_near_urban_station
            push!(all_points, GridPoint(lon, lat, elev_val, emision_grid[idx], vx_grid[idx], vy_grid[idx], grid_pts[1, idx], grid_pts[2, idx], grid_pts[3, idx]))
        end
        idx += 1
    end
    
    # Buscar el pico de emisión (local maximum) en el vecindario de cada estación urbana
    candidates = GridPoint[]
    for st in inputs
        # Excluir estaciones de alta montaña de ser centros de búsqueda de focos urbanos
        if Float64(st["elevacion"]) >= 1800.0
            continue
        end
        st_lon = Float64(st["longitud"])
        st_lat = Float64(st["latitud"])
        
        best_gp = nothing
        max_S = -Inf
        
        for gp in all_points
            dist = sqrt((gp.lon - st_lon)^2 + (gp.lat - st_lat)^2)
            if dist <= 0.022  # Buscar en un radio estrecho de ~2.4 km
                if gp.S > max_S
                    max_S = gp.S
                    best_gp = gp
                end
            end
        end
        
        if best_gp !== nothing
            push!(candidates, best_gp)
        end
    end
    
    # Ordenar candidatos por emisión S descendente
    sort!(candidates, by = p -> p.S, rev=true)
    
    # NMS (Non-maximum suppression) para seleccionar focos de emisión espacialmente separados a partir de los candidatos
    hotspots = GridPoint[]
    min_dist_deg = 0.032 # Aprox 3.5 km de distancia mínima para asegurar que cubran distintas comunas
    for gp in candidates
        too_close = false
        for hs in hotspots
            dist = sqrt((gp.lon - hs.lon)^2 + (gp.lat - hs.lat)^2)
            if dist < min_dist_deg
                too_close = true
                break
            end
        end
        if !too_close
            push!(hotspots, gp)
            println("📍 Foco Detectado: Lon=$(round(gp.lon, digits=4)), Lat=$(round(gp.lat, digits=4)), Emisión S=$(round(gp.S, digits=6))")
        end
        if length(hotspots) >= 8 # Seleccionar los 8 focos principales del valle
            break
        end
    end
    
    # Trazar trayectorias físicas de dispersión saliendo de las fuentes encontradas
    hotspots_data = Dict{String, Any}[]
    for (h_idx, hs) in enumerate(hotspots)
        x = hs.x
        y = hs.y
        z = hs.z
        t = 1.0
        
        path = Vector{Float64}[]
        push!(path, [hs.lon, hs.lat, hs.elev])
        
        for step in 1:15
            pt = reshape([x, y, z, t], 4, 1)
            vx_val = phi[3](pt, theta.depvar.vx)[1]
            vy_val = phi[4](pt, theta.depvar.vy)[1]
            vz_val = phi[5](pt, theta.depvar.vz)[1]
            
            x = clamp(x + vx_val * dt_scaled, -1.0, 1.0)
            y = clamp(y + vy_val * dt_scaled, -1.0, 1.0)
            z = clamp(z + vz_val * dt_scaled, 0.0, 1.0)
            
            lon = lon_min + (x + 1.0)/2.0 * (lon_max - lon_min)
            lat = lat_min + (y + 1.0)/2.0 * (lat_max - lat_min)
            elev = elev_min + z * (elev_max - elev_min)
            push!(path, [lon, lat, elev])
        end
        
        push!(hotspots_data, Dict(
            "id" => h_idx,
            "longitud" => hs.lon,
            "latitud" => hs.lat,
            "elevacion" => hs.elev,
            "emision_S_ug_m3_s" => round(hs.S, digits=6),
            "vx" => round(hs.vx, digits=3),
            "vy" => round(hs.vy, digits=3),
            "trajectory" => path
        ))
    end
    
    sources_path = "output_sources.json"
    open(sources_path, "w") do f
        JSON.print(f, hotspots_data, 4)
    end
    println("✅ ¡Fuentes de emisión inferidas guardadas exitosamente en '$sources_path'!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_prediction()
end
