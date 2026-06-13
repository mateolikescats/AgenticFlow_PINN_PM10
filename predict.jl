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

function is_in_valley_floor(lon::Float64, lat::Float64)
    # Hacemos una interpolación lineal por tramos de la longitud central del valle según la latitud
    center_lon = -75.57 # Valor basal por defecto (Medellín Centro)
    
    if lat < 6.09308
        center_lon = -75.63776
    elseif lat < 6.14550
        # Tramo Caldas -> Sabaneta
        t = (lat - 6.09308) / (6.14550 - 6.09308)
        center_lon = -75.63776 + t * (-75.62126 - -75.63776)
    elseif lat < 6.16868
        # Tramo Sabaneta -> Itagüí
        t = (lat - 6.14550) / (6.16868 - 6.14550)
        center_lon = -75.62126 + t * (-75.58197 - -75.62126)
    elseif lat < 6.22189
        # Tramo Itagüí -> Belén
        t = (lat - 6.16868) / (6.22189 - 6.16868)
        center_lon = -75.58197 + t * (-75.61060 - -75.58197)
    elseif lat < 6.25256
        # Tramo Belén -> Medellín Centro
        t = (lat - 6.22189) / (6.25256 - 6.22189)
        center_lon = -75.61060 + t * (-75.56958 - -75.61060)
    elseif lat < 6.33755
        # Tramo Medellín Centro -> Bello
        t = (lat - 6.25256) / (6.33755 - 6.25256)
        center_lon = -75.56958 + t * (-75.56780 - -75.56958)
    elseif lat < 6.34536
        # Tramo Bello -> Copacabana
        t = (lat - 6.33755) / (6.34536 - 6.33755)
        center_lon = -75.56780 + t * (-75.50475 - -75.56780)
    else
        # Tramo Copacabana -> Girardota
        t = (lat - 6.34536) / (6.43696 - 6.34536)
        center_lon = -75.50475 + t * (-75.33040 - -75.50475)
    end
    
    # Ancho del valle (permitimos un margen de +- 0.024 grados de longitud, unos 2.6 km)
    # Esto contiene el plano del valle pero excluye las laderas montañosas profundas
    return abs(lon - center_lon) <= 0.024
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
            # Multiplicamos por -1.0 para corregir la convención de viento de meteorológica a física
            vx = -phi[3](pt, theta.depvar.vx)[1]
            vy = -phi[4](pt, theta.depvar.vy)[1]
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
        # Multiplicamos por -1.0 para corregir la convención de viento de meteorológica a física
        vx_est = -vx_pred[i] * 10.0
        vy_est = -vy_pred[i] * 10.0
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
    # ==========================================================================
    # CÁLCULO DE FUENTES - DOBLE ENFOQUE (AUTÓNOMO & URBANO)
    # ==========================================================================
    println("\n🔎 Calculando focos de emisión con DOBLE CAPA (Industrial Autónomo y Urbano Tránsito)...")
    
    hotspots_data = Dict{String, Any}[]
    
    # --- PARTE 1: FUENTES URBANAS (TRÁNSITO / CIUDAD) ---
    println("\n1️⃣ Procesando Fuentes Urbanas (Plano del Valle)...")
    urban_centers = [
        (name="Girardota (Norte)", lat=6.4369602, lon=-75.3303986, id=1, elev=1313.0),
        (name="Copacabana (Norte)", lat=6.3453598, lon=-75.5047531, id=2, elev=1455.0),
        (name="Bello (Norte-Centro)", lat=6.3375502, lon=-75.5678024, id=3, elev=1506.0),
        (name="Medellín Centro (Tráfico)", lat=6.2525611, lon=-75.5695801, id=4, elev=1479.0),
        (name="Medellín Belén (Industrial)", lat=6.2218938, lon=-75.6106033, id=5, elev=1582.0),
        (name="Envigado (Sur)", lat=6.1856666, lon=-75.5972061, id=9, elev=1532.0),
        (name="Envigado Ladera (Sur)", lat=6.1825418, lon=-75.5506363, id=10, elev=1950.0),
        (name="Itagüí (Sur Industrial)", lat=6.1686831, lon=-75.5819702, id=6, elev=1584.0),
        (name="Sabaneta (Sur)", lat=6.1455002, lon=-75.6212616, id=7, elev=1626.0),
        (name="Caldas (Sur)", lat=6.0930777, lon=-75.6377640, id=8, elev=1759.0)
    ]
    
    for (idx, uc) in enumerate(urban_centers)
        x_scaled = 2.0 * (uc.lon - lon_min) / (lon_max - lon_min) - 1.0
        y_scaled = 2.0 * (uc.lat - lat_min) / (lat_max - lat_min) - 1.0
        z_scaled = (uc.elev - elev_min) / (elev_max - elev_min)
        t_scaled = 1.0
        
        pt = reshape([x_scaled, y_scaled, z_scaled, t_scaled], 4, 1)
        
        u_val = phi[1](pt, theta.depvar.u)[1]
        pm25_est = clamp(u_val * 100.0, 0.0, Inf)
        
        S_val = phi[7](pt, theta.depvar.S)[1]
        emision_est = S_val * (100.0 / 3600.0)
        
        # Multiplicamos por -1.0 para corregir la convención de viento de meteorológica a física
        vx_est = -phi[3](pt, theta.depvar.vx)[1] * 10.0
        vy_est = -phi[4](pt, theta.depvar.vy)[1] * 10.0
        
        # Trayectoria
        x, y, z, t = x_scaled, y_scaled, z_scaled, 1.0
        path = Vector{Float64}[]
        push!(path, [uc.lon, uc.lat, uc.elev])
        for step in 1:15
            pt_step = reshape([x, y, z, t], 4, 1)
            # Multiplicamos por -1.0 para corregir la convención de viento de meteorológica a física
            vx_val = -phi[3](pt_step, theta.depvar.vx)[1]
            vy_val = -phi[4](pt_step, theta.depvar.vy)[1]
            vz_val = phi[5](pt_step, theta.depvar.vz)[1]
            
            x = clamp(x + vx_val * dt_scaled, -1.0, 1.0)
            y = clamp(y + vy_val * dt_scaled, -1.0, 1.0)
            z = clamp(z + vz_val * dt_scaled, 0.0, 1.0)
            
            push!(path, [
                lon_min + (x + 1.0)/2.0 * (lon_max - lon_min),
                lat_min + (y + 1.0)/2.0 * (lat_max - lat_min),
                elev_min + z * (elev_max - elev_min)
            ])
        end
        
        println("   🚦 Urbano: $(uc.name) | PM2.5=$(round(pm25_est, digits=1)), S=$(round(emision_est, digits=6))")
        
        push!(hotspots_data, Dict(
            "id" => uc.id,
            "name" => uc.name,
            "type" => "urban",
            "longitud" => uc.lon,
            "latitud" => uc.lat,
            "elevacion" => uc.elev,
            "pm25_ug_m3" => round(pm25_est, digits=3),
            "emision_S_ug_m3_s" => round(emision_est, digits=6),
            "vx" => round(vx_est, digits=3),
            "vy" => round(vy_est, digits=3),
            "trajectory" => path
        ))
    end
    
    # --- PARTE 2: FUENTES INDUSTRIALES (AUTÓNOMAS - GRILLA + NMS) ---
    println("\n2️⃣ Procesando Fuentes Industriales Autónomas (Búsqueda en Grilla + NMS)...")
    
    grid_lon = range(lon_min + 0.03, lon_max - 0.03, length=35)
    grid_lat = range(lat_min + 0.03, lat_max - 0.03, length=35)
    N_grid = length(grid_lon) * length(grid_lat)
    
    grid_pts = Matrix{Float64}(undef, 4, N_grid)
    idx = 1
    for lon in grid_lon, lat in grid_lat
        x_scaled = 2.0 * (lon - lon_min) / (lon_max - lon_min) - 1.0
        y_scaled = 2.0 * (lat - lat_min) / (lat_max - lat_min) - 1.0
        
        # Interpolar la elevación real del terreno a partir de las estaciones
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
        
        grid_pts[1, idx] = x_scaled
        grid_pts[2, idx] = y_scaled
        grid_pts[3, idx] = z_scaled
        grid_pts[4, idx] = 1.0
        idx += 1
    end
    
    # Evaluar red de emisión S
    S_grid = phi[7](grid_pts, theta.depvar.S)
    emision_grid = S_grid * (100.0 / 3600.0)
    
    # Evaluar red de concentración u
    u_grid = phi[1](grid_pts, theta.depvar.u) * 100.0
    
    # Evaluar vientos
    vx_grid = phi[3](grid_pts, theta.depvar.vx) * 10.0
    vy_grid = phi[4](grid_pts, theta.depvar.vy) * 10.0
    
    # Estructurar puntos filtrando por cercanía a estaciones (huella del sensor)
    # para evitar extrapolar en los límites extremos del recuadro
    all_grid_points = GridPoint[]
    idx = 1
    for lon in grid_lon, lat in grid_lat
        is_near_sensors = false
        for st in inputs
            dist = sqrt((lon - Float64(st["longitud"]))^2 + (lat - Float64(st["latitud"]))^2)
            if dist <= 0.06  # Radio de 6.6 km alrededor de cualquier estación
                is_near_sensors = true
                break
            end
        end
        
        if is_near_sensors
            # Re-calcular elevación interpolada
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
            
            push!(all_grid_points, GridPoint(
                lon, lat, elev_val, 
                emision_grid[idx], vx_grid[idx], vy_grid[idx], 
                grid_pts[1, idx], grid_pts[2, idx], grid_pts[3, idx]
            ))
        end
        idx += 1
    end
    
    # 1. Filtrar candidatos usando el perfil geográfico del plano del valle (offline-safe y robusto)
    valley_candidates = GridPoint[]
    for gp in all_grid_points
        if is_in_valley_floor(gp.lon, gp.lat)
            push!(valley_candidates, gp)
        end
    end

    # 2. Ordenar candidatos del valle por emisión S descendente
    sort!(valley_candidates, by = p -> p.S, rev=true)
    
    # 3. Tomar los mejores 60 candidatos del plano del valle para optimizar continuamente (Hill-Climbing)
    # y así obtener coordenadas totalmente continuas libres de rejilla (mesh-free)
    n_opt = min(60, length(valley_candidates))
    refined_candidates = GridPoint[]
    
    for gp in valley_candidates[1:n_opt]
        x_opt, y_opt, z_opt = gp.x, gp.y, gp.z
        step_x = 0.005
        step_y = 0.005
        step_z = 0.005
        
        for iter in 1:25
            pt_cur = reshape([x_opt, y_opt, z_opt, 1.0], 4, 1)
            best_S = phi[7](pt_cur, theta.depvar.S)[1]
            improved = false
            
            for (dx, dy, dz) in [
                (step_x, 0.0, 0.0), (-step_x, 0.0, 0.0),
                (0.0, step_y, 0.0), (0.0, -step_y, 0.0),
                (0.0, 0.0, step_z), (0.0, 0.0, -step_z)
            ]
                xn = clamp(x_opt + dx, -1.0, 1.0)
                yn = clamp(y_opt + dy, -1.0, 1.0)
                zn = clamp(z_opt + dz, 0.0, 1.0)
                
                lon_n = lon_min + (xn + 1.0)/2.0 * (lon_max - lon_min)
                lat_n = lat_min + (yn + 1.0)/2.0 * (lat_max - lat_min)
                
                # Mantener la optimización acotada estrictamente dentro del plano del valle
                if !is_in_valley_floor(lon_n, lat_n)
                    continue
                end
                
                pt_n = reshape([xn, yn, zn, 1.0], 4, 1)
                S_n = phi[7](pt_n, theta.depvar.S)[1]
                
                if S_n > best_S
                    x_opt, y_opt, z_opt = xn, yn, zn
                    improved = true
                    break
                end
            end
            if !improved
                step_x *= 0.5
                step_y *= 0.5
                step_z *= 0.5
            end
            if step_x < 1e-6
                break
            end
        end
        
        # Convertir a coordenadas físicas continuas refinadas
        lon_opt = lon_min + (x_opt + 1.0)/2.0 * (lon_max - lon_min)
        lat_opt = lat_min + (y_opt + 1.0)/2.0 * (lat_max - lat_min)
        
        # Interpolar elevación para la nueva coordenada continua
        w_sum = 0.0
        elev_sum = 0.0
        for st in inputs
            dist = sqrt((lon_opt - Float64(st["longitud"]))^2 + (lat_opt - Float64(st["latitud"]))^2)
            if dist < 1e-5
                elev_sum = Float64(st["elevacion"])
                w_sum = 1.0
                break
            end
            w = 1.0 / (dist^2)
            w_sum += w
            elev_sum += Float64(st["elevacion"]) * w
        end
        elev_opt = elev_sum / w_sum
        
        # Evaluar emisión y viento refinados continuamente
        pt_opt = reshape([x_opt, y_opt, z_opt, 1.0], 4, 1)
        S_opt = phi[7](pt_opt, theta.depvar.S)[1] * (100.0 / 3600.0)
        vx_opt = phi[3](pt_opt, theta.depvar.vx)[1] * 10.0
        vy_opt = phi[4](pt_opt, theta.depvar.vy)[1] * 10.0
        
        push!(refined_candidates, GridPoint(
            lon_opt, lat_opt, elev_opt,
            S_opt, vx_opt, vy_opt,
            x_opt, y_opt, z_opt
        ))
    end
    
    # 4. Ordenar los candidatos refinados por su emisión S óptima
    sort!(refined_candidates, by = p -> p.S, rev=true)
    
    # 5. Ejecutar NMS espacial libre sobre coordenadas continuas refinadas
    hotspots = GridPoint[]
    min_dist_deg = 0.032 # Aprox 3.5 km de distancia mínima
    for gp in refined_candidates
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
        end
        if length(hotspots) >= 8
            break
        end
    end
    
    # Agregar las fuentes industriales encontradas
    for (h_idx, hs) in enumerate(hotspots)
        # Buscar el nombre del municipio más cercano para darle un identificador legible
        closest_name = "Zona Ladera"
        min_d = Inf
        for st in inputs
            dist = sqrt((hs.lon - Float64(st["longitud"]))^2 + (hs.lat - Float64(st["latitud"]))^2)
            if dist < min_d
                min_d = dist
                # Buscar correspondencia de nombre
                matched_name = "Zona Ladera"
                for uc in urban_centers
                    d_uc = sqrt((uc.lon - Float64(st["longitud"]))^2 + (uc.lat - Float64(st["latitud"]))^2)
                    if d_uc < 1e-4
                        matched_name = "Ladera de " * uc.name
                        break
                    end
                end
                closest_name = matched_name
            end
        end
        if closest_name == "Zona Ladera"
            closest_name = "Foco Autónomo #" * string(h_idx)
        end
        
        # Consultar la concentración exacta en el hotspot
        pt = reshape([hs.x, hs.y, hs.z, 1.0], 4, 1)
        u_val = phi[1](pt, theta.depvar.u)[1]
        pm25_est = clamp(u_val * 100.0, 0.0, Inf)
        
        # Trayectoria
        x, y, z, t = hs.x, hs.y, hs.z, 1.0
        path = Vector{Float64}[]
        push!(path, [hs.lon, hs.lat, hs.elev])
        for step in 1:15
            pt_step = reshape([x, y, z, t], 4, 1)
            # Multiplicamos por -1.0 para corregir la convención de viento de meteorológica a física
            vx_val = -phi[3](pt_step, theta.depvar.vx)[1]
            vy_val = -phi[4](pt_step, theta.depvar.vy)[1]
            vz_val = phi[5](pt_step, theta.depvar.vz)[1]
            
            x = clamp(x + vx_val * dt_scaled, -1.0, 1.0)
            y = clamp(y + vy_val * dt_scaled, -1.0, 1.0)
            z = clamp(z + vz_val * dt_scaled, 0.0, 1.0)
            
            push!(path, [
                lon_min + (x + 1.0)/2.0 * (lon_max - lon_min),
                lat_min + (y + 1.0)/2.0 * (lat_max - lat_min),
                elev_min + z * (elev_max - elev_min)
            ])
        end
        
        println("   🏭 Industrial Autónomo: $(closest_name) | PM2.5=$(round(pm25_est, digits=1)), S=$(round(hs.S, digits=6))")
        
        push!(hotspots_data, Dict(
            "id" => Int(round(hs.S * 1000000.0)) + h_idx * 100, # ID único que no colisione
            "name" => closest_name,
            "type" => "industrial",
            "longitud" => hs.lon,
            "latitud" => hs.lat,
            "elevacion" => hs.elev,
            "pm25_ug_m3" => round(pm25_est, digits=3),
            "emision_S_ug_m3_s" => round(hs.S, digits=6),
            "vx" => round(-hs.vx, digits=3), # Negado
            "vy" => round(-hs.vy, digits=3), # Negado
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
