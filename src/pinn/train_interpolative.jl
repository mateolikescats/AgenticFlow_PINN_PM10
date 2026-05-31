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

function load_data(filepath)
    if !isfile(filepath)
        println("⚠️ Archivo de datos no encontrado: $filepath. Usando datos ficticios 3D acoplados (x, y, z, t, u, T, vx, vy).")
        return [
            Dict("id" => "C-001", "x" => -0.5, "y" => -0.5, "z" => 0.1, "t" => 0.0, "u" => 0.3, "T" => -0.8, "vx" => 0.2, "vy" => -0.1, "elevacion_real" => 1550.0, "pm25" => 30.0),
            Dict("id" => "C-002", "x" => 0.0, "y" => 0.0, "z" => 0.0, "t" => 0.5, "u" => 0.5, "T" => -0.5, "vx" => 0.1, "vy" => 0.1, "elevacion_real" => 1500.0, "pm25" => 50.0),
            Dict("id" => "C-003", "x" => 0.5, "y" => 0.5, "z" => 0.2, "t" => 1.0, "u" => 0.2, "T" => -0.2, "vx" => -0.1, "vy" => 0.3, "elevacion_real" => 1700.0, "pm25" => 20.0)
        ]
    end
    return JSON.parsefile(filepath)
end

function train_interpolative(data_path::String="datos_siata_temporal.json")
    println("==== Iniciando Fase Interpolativa PINN Termodinámica ====")

    # Leer hiperparámetros si existen (inyectados por el Agente Python)
    epochs = 10000
    learning_rate = 0.025
    lbfgs_iters = 3000
    if isfile("pinn_config.json")
        try
            config = JSON.parsefile("pinn_config.json")
            epochs = get(config, "epochs", 100)
            learning_rate = get(config, "learning_rate", 0.01)
            lbfgs_iters = get(config, "lbfgs_iters", 300)
            println("Configuración recibida del Agente: Epochs=$epochs, LR=$learning_rate, L-BFGS-Iters=$lbfgs_iters")
        catch
            println("Error leyendo pinn_config.json, usando valores por defecto.")
        end
    end

    # 1. Obtener la ecuación Boussinesq y arquitecturas 3D (7 redes)
    pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S) = get_boussinesq_pde_system()
    chains = build_multi_pinn()

    # 2. Cargar datos empíricos (PM2.5)
    data = load_data(data_path)
    
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

    # Extraer variables de entrenamiento
    x_data = Float64[get(d, "x", 0.0) for d in train_data]
    y_data = Float64[get(d, "y", 0.0) for d in train_data]
    z_data = Float64[get(d, "z", 0.0) for d in train_data]
    t_data = Float64[get(d, "t", 0.0) for d in train_data]
    u_data = Float64[get(d, "u", 0.0) for d in train_data]
    T_data = Float64[get(d, "T", 0.0) for d in train_data]

    # Extraer variables de validación
    val_x_data = Float64[get(d, "x", 0.0) for d in val_data]
    val_y_data = Float64[get(d, "y", 0.0) for d in val_data]
    val_z_data = Float64[get(d, "z", 0.0) for d in val_data]
    val_t_data = Float64[get(d, "t", 0.0) for d in val_data]
    val_u_data = Float64[get(d, "u", 0.0) for d in val_data]
    val_T_data = Float64[get(d, "T", 0.0) for d in val_data]

    # Definir la función de pérdida de validación para la concentración de PM2.5 (phi[1])
    eval_validation_loss = (phi, θ) -> begin
        loss_val = 0.0
        n_points = length(val_u_data)
        if n_points == 0
            return 0.0
        end
        phi_u = phi[1]
        for i in 1:n_points
            coords = [val_x_data[i], val_y_data[i], val_z_data[i], val_t_data[i]]
            pred_u = phi_u(coords, θ.depvar.u)[1]
            loss_val += (pred_u - val_u_data[i])^2
        end
        return loss_val / n_points
    end

    # Datos empíricos (Meteorología - Viento)
    meteo_data = load_data("datos_meteorologicos_viento.json")
    x_meteo = Float64[get(d, "x", 0.0) for d in meteo_data]
    y_meteo = Float64[get(d, "y", 0.0) for d in meteo_data]
    z_meteo = Float64[get(d, "z", 0.0) for d in meteo_data]
    t_meteo = Float64[get(d, "t", 0.0) for d in meteo_data]
    vx_meteo = Float64[get(d, "vx", 0.0) for d in meteo_data]
    vy_meteo = Float64[get(d, "vy", 0.0) for d in meteo_data]

    # 3. Función de Pérdida Adicional (Data Loss)
    # phi es una tupla de funciones, una por cada red: [phi_u, phi_T, phi_vx, phi_vy, phi_vz, phi_P, phi_S]
    additional_loss = (phi, θ, p) -> begin
        loss_u = 0.0
        loss_T = 0.0
        n_points = length(u_data)

        phi_u = phi[1] # Red para la concentración u
        phi_T = phi[2] # Red para la temperatura T
        phi_vx = phi[3] # Red para viento transversal
        phi_vy = phi[4] # Red para viento longitudinal

        # Pérdida de PM2.5 y Temperatura
        for i in 1:n_points
            coords = [x_data[i], y_data[i], z_data[i], t_data[i]]
            pred_u = phi_u(coords, θ.depvar.u)[1]
            pred_T = phi_T(coords, θ.depvar.T)[1]

            loss_u += (pred_u - u_data[i])^2
            loss_T += (pred_T - T_data[i])^2
        end

        # Pérdida de Asimilación Meteorológica (Velocidades)
        loss_v = 0.0
        n_meteo = length(vx_meteo)
        for i in 1:n_meteo
            coords = [x_meteo[i], y_meteo[i], z_meteo[i], t_meteo[i]]
            pred_vx = phi_vx(coords, θ.depvar.vx)[1]
            pred_vy = phi_vy(coords, θ.depvar.vy)[1]

            loss_v += (pred_vx - vx_meteo[i])^2 + (pred_vy - vy_meteo[i])^2
        end

        return (loss_u / max(n_points, 1)) + (loss_T / max(n_points, 1)) + (loss_v / max(n_meteo, 1))
    end

    # 4. Estrategia de Discretización (Physics-Informed) con Muestreo de Importancia (Propuesta 3)
    strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())

    # 5. Configurar el discretizador de NeuralPDE usando Pesos Adaptativos (Adaptive Weights)
    # Esto equilibra dinámicamente las pérdidas de datos (u, T) vs los residuales físicos (Navier-Stokes)
    adaptive_strategy = GradientScaleAdaptiveLoss(100) # Actualiza los pesos cada 100 epochs

    discretization = PhysicsInformedNN(chains, strategy;
        additional_loss=additional_loss,
        weight_strategy=adaptive_strategy)

    # 6. Convertir PDESystem al problema de Optimización
    println("Compilando el problema Boussinesq (esto tomará tiempo por las 5 ecuaciones)...")
    prob = discretize(pdesys, discretization)

    # 7. Ciclo de Entrenamiento
    epoch_count = 0
    callback_adam = function (p, l, args...)
        epoch_count += 1
        val_loss = 0.0
        try
            val_loss = eval_validation_loss(discretization.phi, p.u)
        catch e
        end
        println("[EPOCH_LOG] Epoch: $epoch_count | Loss: $l | Val Loss: $val_loss")
        return false
    end

    println("Fase 1: Entrenando con Adam (LR=$learning_rate)...")
    res1 = Optimization.solve(prob, OptimizationOptimisers.Adam(learning_rate); callback=callback_adam, maxiters=epochs)
    println("Fase 1 terminada. Loss Adam: ", res1.objective)

    # Fase 2: Refinamiento de precisión con L-BFGS (Optimizador de segundo orden)
    callback_lbfgs = function (p, l, args...)
        epoch_count += 1
        val_loss = 0.0
        try
            val_loss = eval_validation_loss(discretization.phi, p.u)
        catch e
        end
        println("[EPOCH_LOG] Epoch: $epoch_count | Loss: $l | Val Loss: $val_loss | Stage: L-BFGS")
        return false
    end

    println("Fase 2: Refinando con L-BFGS...")
    prob2 = Optimization.remake(prob, u0=res1.u)
    res2 = Optimization.solve(prob2, OptimizationOptimJL.LBFGS(); callback=callback_lbfgs, maxiters=lbfgs_iters)
    println("Fase 2 terminada. Loss final L-BFGS: ", res2.objective)

    println("Exportando metadatos y pesos reales acoplados...")
    open("pesos_pinn_boussinesq.json", "w") do f
        JSON.print(f, Dict("loss" => res2.objective, "info" => "Pesos exportados en formato binario JLD2."))
    end

    # Exportar los pesos reales de las 6 redes usando JLD2 para la posterior Inferencia
    @save "modelo_pinn.jld2" theta = res2.u

    println("¡Entrenamiento y modelo exportados exitosamente!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_interpolative()
end
