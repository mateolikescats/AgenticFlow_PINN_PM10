using Pkg
Pkg.activate(".")

include("AdvectionDiffusion.jl")
using .AdvectionDiffusion

using NeuralPDE
using ModelingToolkit
using Lux
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using ComponentArrays
using JSON
using JLD2 # Para guardar los pesos reales
using Random
using LineSearches

function clean_wind_id(id_str::String)
    clean_str = replace(id_str, "W-" => "")
    return round(Int, parse(Float64, clean_str))
end

function load_data(filepath)
    if !isfile(filepath)
        error("❌ ERROR: El archivo de datos requerido '$filepath' no existe.")
    end
    return JSON.parsefile(filepath)
end

function load_processed_data(pm_path::String, wind_path::String)
    # 1. Cargar datos de viento y construir diccionario de coordenadas de estaciones
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

    # 2. Cargar datos crudos de PM2.5 y eliminar duplicados/no coincidentes
    pm_raw = load_data(pm_path)
    seen_records = Set{Tuple{Int, Float64}}()
    cleaned_pm = []

    for d in pm_raw
        id = round(Int, d["id"])
        t_val = Float64(d["timestamp"])
        pm_val = Float64(d["pm25"])
        
        # Evitar duplicados por estación y marca de tiempo
        rec_key = (id, t_val)
        if rec_key in seen_records
            continue
        end
        
        # Validar si la estación existe en la meteorología
        if !haskey(station_coords, id)
            continue
        end
        
        push!(seen_records, rec_key)
        push!(cleaned_pm, Dict("id" => id, "timestamp" => t_val, "pm25" => pm_val))
    end

    # Ordenar cronológicamente
    sort!(cleaned_pm, by = d -> (d["timestamp"], d["id"]))

    # 3. Adimensionalizar y generar perfil térmico
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

    println("✅ Carga y alineación de datos de PM2.5 exitosa: ", length(final_pm_data), " registros procesados.")
    return final_pm_data
end

function train_interpolative(pm_path::String="data/datos_oficiales_pm25.json", wind_path::String="data/datos_meteorologicos_viento.json")
    println("==== Iniciando Fase Interpolativa PINN Termodinámica ====")
    
    # Leer hiperparámetros si existen (inyectados por el Agente Python)
    epochs = 10000
    learning_rate = 0.025
    lbfgs_iters = 3000
    if isfile("models/pinn_config.json")
        try
            config = JSON.parsefile("models/pinn_config.json")
            epochs = get(config, "epochs", 100)
            learning_rate = get(config, "learning_rate", 0.01)
            lbfgs_iters = get(config, "lbfgs_iters", 300)
            println("Configuración recibida del Agente: Epochs=$epochs, LR=$learning_rate, L-BFGS-Iters=$lbfgs_iters")
        catch
            println("Error leyendo models/pinn_config.json, usando valores por defecto.")
        end
    end

    # 1. Obtener la ecuación Boussinesq y arquitecturas 3D (7 redes)
    pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S) = get_boussinesq_pde_system()
    chains = build_multi_pinn()

    # 2. Cargar y procesar datos empíricos (PM2.5) alineados en memoria
    data = load_processed_data(pm_path, wind_path)
    
    # Mitigación de Spatial Data Leakage: División 80/20 basada en estaciones físicas
    station_ids = unique(String[string(get(d, "id", "unknown")) for d in data])
    Random.seed!(42)
    shuffled_ids = shuffle(station_ids)
    n_train = max(1, round(Int, 0.8 * length(shuffled_ids)))
    train_station_ids = shuffled_ids[1:n_train]
    val_station_ids = shuffled_ids[n_train+1:end]

    train_data = filter(d -> string(get(d, "id", "unknown")) in train_station_ids, data)
    val_data = filter(d -> string(get(d, "id", "unknown")) in val_station_ids, data)
    if isempty(val_data) && length(data) > 1
        n_rows = length(data)
        n_train_rows = max(1, round(Int, 0.8 * n_rows))
        train_data = data[1:n_train_rows]
        val_data = data[n_train_rows+1:end]
    end

    # Optimización: Reducir tamaño de datos para evitar bucles gigantescos en entrenamiento
    if length(train_data) > 5000
        Random.seed!(42)
        train_data = shuffle(train_data)[1:5000]
    end
    if length(val_data) > 1000
        Random.seed!(42)
        val_data = shuffle(val_data)[1:1000]
    end

    # Extraer variables de entrenamiento
    x_data = Float64[get(d, "x", 0.0) for d in train_data]
    y_data = Float64[get(d, "y", 0.0) for d in train_data]
    z_data = Float64[get(d, "z", 0.0) for d in train_data]
    t_data = Float64[get(d, "t", 0.0) for d in train_data]
    u_data = Float64[get(d, "u", 0.0) for d in train_data]
    T_data = Float64[get(d, "T", 0.0) for d in train_data]

    # Crear matrices de coordenadas para evaluación vectorizada (Lux espera [4, N])
    train_coords = hcat([[x_data[i], y_data[i], z_data[i], t_data[i]] for i in 1:length(x_data)]...)
    u_data_vec = reshape(u_data, 1, :)
    T_data_vec = reshape(T_data, 1, :)

    # Extraer variables de validación
    val_x_data = Float64[get(d, "x", 0.0) for d in val_data]
    val_y_data = Float64[get(d, "y", 0.0) for d in val_data]
    val_z_data = Float64[get(d, "z", 0.0) for d in val_data]
    val_t_data = Float64[get(d, "t", 0.0) for d in val_data]
    val_u_data = Float64[get(d, "u", 0.0) for d in val_data]
    val_T_data = Float64[get(d, "T", 0.0) for d in val_data]

    val_coords = hcat([[val_x_data[i], val_y_data[i], val_z_data[i], val_t_data[i]] for i in 1:length(val_x_data)]...)
    val_u_data_vec = reshape(val_u_data, 1, :)
    val_T_data_vec = reshape(val_T_data, 1, :)

    # 3. Cargar y split de Datos de Viento (Meteorología)
    meteo_data_all = load_data(wind_path)
    
    # Mitigación de Spatial Data Leakage (Viento): División 80/20 basada en estaciones meteorológicas
    wind_station_ids = unique(String[string(get(d, "id", "unknown")) for d in meteo_data_all])
    Random.seed!(42)
    shuffled_wind_ids = shuffle(wind_station_ids)
    n_wind_train = max(1, round(Int, 0.8 * length(shuffled_wind_ids)))
    train_wind_station_ids = shuffled_wind_ids[1:n_wind_train]
    val_wind_station_ids = shuffled_wind_ids[n_wind_train+1:end]

    train_wind_data = filter(d -> string(get(d, "id", "unknown")) in train_wind_station_ids, meteo_data_all)
    val_wind_data = filter(d -> string(get(d, "id", "unknown")) in val_wind_station_ids, meteo_data_all)
    if isempty(val_wind_data) && length(meteo_data_all) > 1
        n_wind_rows = length(meteo_data_all)
        n_wind_train_rows = max(1, round(Int, 0.8 * n_wind_rows))
        train_wind_data = meteo_data_all[1:n_wind_train_rows]
        val_wind_data = meteo_data_all[n_wind_train_rows+1:end]
    end

    # Extraer variables de entrenamiento de viento
    x_meteo = Float64[get(d, "x", 0.0) for d in train_wind_data]
    y_meteo = Float64[get(d, "y", 0.0) for d in train_wind_data]
    z_meteo = Float64[get(d, "z", 0.0) for d in train_wind_data]
    t_meteo = Float64[get(d, "t", 0.0) for d in train_wind_data]
    vx_meteo = Float64[get(d, "vx", 0.0) for d in train_wind_data]
    vy_meteo = Float64[get(d, "vy", 0.0) for d in train_wind_data]

    meteo_coords = hcat([[x_meteo[i], y_meteo[i], z_meteo[i], t_meteo[i]] for i in 1:length(x_meteo)]...)
    vx_meteo_vec = reshape(vx_meteo, 1, :)
    vy_meteo_vec = reshape(vy_meteo, 1, :)

    # Extraer variables de validación de viento
    val_x_meteo = Float64[get(d, "x", 0.0) for d in val_wind_data]
    val_y_meteo = Float64[get(d, "y", 0.0) for d in val_wind_data]
    val_z_meteo = Float64[get(d, "z", 0.0) for d in val_wind_data]
    val_t_meteo = Float64[get(d, "t", 0.0) for d in val_wind_data]
    val_vx_meteo = Float64[get(d, "vx", 0.0) for d in val_wind_data]
    val_vy_meteo = Float64[get(d, "vy", 0.0) for d in val_wind_data]

    val_meteo_coords = hcat([[val_x_meteo[i], val_y_meteo[i], val_z_meteo[i], val_t_meteo[i]] for i in 1:length(val_x_meteo)]...)
    val_vx_meteo_vec = reshape(val_vx_meteo, 1, :)
    val_vy_meteo_vec = reshape(val_vy_meteo, 1, :)

    # Definir la función de pérdida de validación multivariable (retorna pérdida agregada y desglose)
    eval_validation_loss = (phi, θ) -> begin
        # 1. Validación de PM2.5
        loss_val_u = 0.0
        if length(val_u_data) > 0
            phi_u = phi[1]
            pred_u = phi_u(val_coords, θ.depvar.u)
            loss_val_u = sum((pred_u .- val_u_data_vec).^2) / length(val_u_data)
        end

        # 2. Validación de Temperatura
        loss_val_T = 0.0
        if length(val_T_data) > 0
            phi_T = phi[2]
            pred_T = phi_T(val_coords, θ.depvar.T)
            loss_val_T = sum((pred_T .- val_T_data_vec).^2) / length(val_T_data)
        end

        # 3. Validación de Viento (Meteorología)
        loss_val_v = 0.0
        if length(val_vx_meteo) > 0
            phi_vx = phi[3]
            phi_vy = phi[4]
            pred_vx = phi_vx(val_meteo_coords, θ.depvar.vx)
            pred_vy = phi_vy(val_meteo_coords, θ.depvar.vy)
            loss_val_v = (sum((pred_vx .- val_vx_meteo_vec).^2) + sum((pred_vy .- val_vy_meteo_vec).^2)) / length(val_vx_meteo)
        end

        total_val_loss = loss_val_u + loss_val_T + loss_val_v
        return total_val_loss, loss_val_u, loss_val_T, loss_val_v
    end

    # 3. Función de Pérdida Adicional (Data Loss) vectorizada
    # phi es una tupla de funciones, una por cada red: [phi_u, phi_T, phi_vx, phi_vy, phi_vz, phi_P, phi_S]
    additional_loss = (phi, θ, p) -> begin
        phi_u = phi[1]  # Red para la concentración u
        phi_T = phi[2]  # Red para la temperatura T
        phi_vx = phi[3] # Red para viento transversal
        phi_vy = phi[4] # Red para viento longitudinal

        # Pérdida de PM2.5 y Temperatura vectorizada en una sola operación de matriz
        pred_u = phi_u(train_coords, θ.depvar.u)
        pred_T = phi_T(train_coords, θ.depvar.T)

        loss_u = sum((pred_u .- u_data_vec).^2) / length(u_data)
        loss_T = sum((pred_T .- T_data_vec).^2) / length(T_data)

        # Pérdida de Asimilación Meteorológica vectorizada
        pred_vx = phi_vx(meteo_coords, θ.depvar.vx)
        pred_vy = phi_vy(meteo_coords, θ.depvar.vy)

        loss_v = (sum((pred_vx .- vx_meteo_vec).^2) + sum((pred_vy .- vy_meteo_vec).^2)) / length(vx_meteo)

        return loss_u + loss_T + loss_v
    end

    # 4. Estrategia de Discretización (Physics-Informed) con Muestreo de Importancia (Propuesta 3)
    strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())

    # 5. Configurar el discretizador de NeuralPDE usando Pesos Adaptativos (Adaptive Weights)
    # Esto equilibra dinámicamente las pérdidas de datos (u, T) vs los residuales físicos (Navier-Stokes)
    adaptive_strategy = GradientScaleAdaptiveLoss(100) # Actualiza los pesos cada 100 epochs

    discretization = PhysicsInformedNN(chains, strategy;
        additional_loss=additional_loss,
        weight_strategy=adaptive_strategy)

    # 6. Convertir PDESystem al problema de Optimización y obtener representación simbólica
    println("Compilando el problema Boussinesq (esto tomará tiempo por las 5 ecuaciones)...")
    prob = discretize(pdesys, discretization)
    println("Compilando representación simbólica para extracción de pérdidas...")
    sym_prob = NeuralPDE.symbolic_discretize(pdesys, discretization)

    # 6b. Manejo de checkpoints para reanudación de entrenamiento
    checkpoint_file = abspath("models/modelo_pinn_checkpoint.jld2")
    u0 = prob.u0
    epoch_count = 0
    if isfile(checkpoint_file)
        println("🔄 Encontrado archivo de checkpoint en $checkpoint_file. Reanudando entrenamiento...")
        try
            @load checkpoint_file theta epoch
            u0 = theta
            epoch_count = epoch
            println("✅ Pesos y época ($epoch_count) cargados exitosamente.")
        catch e
            println("⚠️ Error cargando checkpoint: ", e)
        end
    else
        println("🆕 No se encontró checkpoint. Iniciando desde parámetros aleatorios.")
    end
    prob = Optimization.remake(prob, u0=u0)

    # Crear/limpiar el archivo de historial de pérdidas (solo si no hay checkpoint)
    hist_file = abspath("data/historial_perdidas.txt")
    if epoch_count == 0
        try
            open(hist_file, "w") do f
                write(f, "epoch,total_loss,pde_loss,bc_loss,data_loss,val_loss,val_u,val_T,val_v,stage\n")
            end
        catch e
            println("⚠️ No se pudo crear data/historial_perdidas.txt: ", e)
        end
    end

    # 7. Ciclo de Entrenamiento
    
    # Función auxiliar para evaluar y registrar pérdidas
    log_losses = (epoch, theta, l, stage; force=false) -> begin
        if force || epoch == 1 || epoch % 50 == 0
            val_loss, val_u, val_T, val_v = 0.0, 0.0, 0.0, 0.0
            try
                val_loss, val_u, val_T, val_v = eval_validation_loss(discretization.phi, theta)
            catch e
            end
            
            pde_loss = 0.0
            bc_loss = 0.0
            try
                pde_loss = sum([f(theta) for f in sym_prob.loss_functions.pde_loss_functions])
                bc_loss = sum([f(theta) for f in sym_prob.loss_functions.bc_loss_functions])
            catch e
            end
            
            data_loss = 0.0
            try
                data_loss = additional_loss(discretization.phi, theta, nothing)
            catch e
            end
            
            try
                open(hist_file, "a") do f
                    write(f, "$epoch,$l,$pde_loss,$bc_loss,$data_loss,$val_loss,$val_u,$val_T,$val_v,$stage\n")
                end
            catch e
                println("⚠️ Error escribiendo historial: ", e)
            end

            # Guardar checkpoint cada 5000 épocas
            if !force && epoch % 5000 == 0
                try
                    @save checkpoint_file theta epoch
                    println("[CHECKPOINT] Checkpoint guardado con éxito en la época $epoch.")
                catch e
                    println("⚠️ Error al guardar checkpoint: ", e)
                end
            end
            
            println("[EPOCH_LOG] Epoch: $epoch | Loss: $l | Val Loss: $val_loss (u: $val_u, T: $val_T, v: $val_v) | PDE: $pde_loss | Data: $data_loss | Stage: $stage")
        end
    end

    callback_adam = function (p, l, args...)
        epoch_count += 1
        log_losses(epoch_count, p.u, l, "Adam")
        return false
    end

    println("Fase 1: Entrenando con Adam (LR=$learning_rate)...")
    res1 = Optimization.solve(prob, OptimizationOptimisers.Adam(learning_rate); callback=callback_adam, maxiters=epochs)
    println("Fase 1 terminada. Loss Adam: ", res1.objective)
    
    # Registrar última época de Adam
    log_losses(epoch_count, res1.u, res1.objective, "Adam-Final"; force=true)

    # Fase 2: Refinamiento de precisión con L-BFGS (Optimizador de segundo orden)
    res2 = res1
    if lbfgs_iters > 0
        callback_lbfgs = function (p, l, args...)
            epoch_count += 1
            log_losses(epoch_count, p.u, l, "L-BFGS")
            return false
        end

        println("Fase 2: Refinando con L-BFGS (usando BackTracking)...")
        prob2 = Optimization.remake(prob, u0=res1.u)
        res2 = Optimization.solve(prob2, OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()); callback=callback_lbfgs, maxiters=lbfgs_iters)
        println("Fase 2 terminada. Loss final L-BFGS: ", res2.objective)
        
        # Registrar última época de L-BFGS
        log_losses(epoch_count, res2.u, res2.objective, "L-BFGS-Final"; force=true)
    else
        println("Fase 2 (L-BFGS) omitida ya que lbfgs_iters = 0.")
    end

    println("Exportando metadatos y pesos reales acoplados...")
    open("models/pesos_pinn_boussinesq.json", "w") do f
        JSON.print(f, Dict("loss" => res2.objective, "info" => "Pesos exportados en formato binario JLD2."))
    end

    # Exportar los pesos reales de las 7 redes usando JLD2 para la posterior Inferencia
    @save "models/modelo_pinn.jld2" theta = res2.u

    println("¡Entrenamiento y modelo exportados exitosamente!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_interpolative()
end
