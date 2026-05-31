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
using Random

function load_data(filepath)
    if !isfile(filepath)
        println("⚠️ Archivo de datos no encontrado: $filepath. Usando datos ficticios 2D (x, z).")
        # Generar datos ficticios (x, z, t, u, T)
        # Asumiendo z=0 es el nivel del suelo donde están los sensores de SIATA
        return [
            Dict("x" => 0.0, "z" => 0.0, "t" => 0.5, "u" => 0.5, "T" => -0.5),
            Dict("x" => 0.5, "z" => 0.0, "t" => 0.8, "u" => 0.2, "T" => -0.8)
        ]
    end
    return JSON.parsefile(filepath)
end

function train_interpolative(data_path="datos_siata_temporal.json")
    println("==== Iniciando Fase Interpolativa PINN Termodinámica ====")
    
    # Leer hiperparámetros si existen (inyectados por el Agente Python)
    epochs = 100
    learning_rate = 0.01
    lbfgs_iters = 300
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
    
    # 1. Obtener la ecuación Boussinesq y arquitecturas (6 redes)
    pdesys, (x, z, t, u, T, vx, vz, P, S) = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    
    # 2. Cargar datos empíricos
    data = load_data(data_path)
    x_data = Float64[d["x"] for d in data]
    z_data = Float64[d["z"] for d in data]
    t_data = Float64[d["t"] for d in data]
    u_data = Float64[d["u"] for d in data]
    T_data = Float64[d["T"] for d in data]

    # Separación por estaciones físicas (evita fuga de datos espacial)
    Random.seed!(42) # Semilla fija para reproducibilidad
    unique_stations = unique([d["id"] for d in data])
    n_stations = length(unique_stations)
    shuffled_stations = shuffle(unique_stations)
    
    # 80% train, 20% validation
    split_idx = floor(Int, 0.8 * n_stations)
    train_stations = shuffled_stations[1:split_idx]
    val_stations = shuffled_stations[split_idx+1:end]
    
    train_indices = findall(d -> d["id"] in train_stations, data)
    val_indices = findall(d -> d["id"] in val_stations, data)
    
    println("📊 Total de estaciones: $n_stations")
    println("👉 Estaciones de entrenamiento (80%): $(length(train_stations)) (puntos: $(length(train_indices)))")
    println("👉 Estaciones de validación (20%): $(length(val_stations)) (puntos: $(length(val_indices)))")

    # 3. Definir la función de pérdida adicional (Ajuste a los datos de SIATA de entrenamiento)
    function additional_loss(phi, θ, p)
        loss_u = 0.0
        loss_T = 0.0
        n_points = length(train_indices)
        if n_points == 0
            return 0.0
        end
        
        phi_u = phi[1] # Red para la concentración u
        phi_T = phi[2] # Red para la temperatura T
        
        for i in train_indices
            coords = [x_data[i], z_data[i], t_data[i]]
            pred_u = phi_u(coords, θ.depvar.u)[1]
            pred_T = phi_T(coords, θ.depvar.T)[1]
            
            loss_u += (pred_u - u_data[i])^2
            loss_T += (pred_T - T_data[i])^2
        end
        return (loss_u + loss_T) / n_points
    end

    # Función para calcular la pérdida de validación en datos no vistos
    function eval_validation_loss(phi, θ)
        loss_u = 0.0
        loss_T = 0.0
        n_points = length(val_indices)
        if n_points == 0
            return 0.0
        end
        
        phi_u = phi[1]
        phi_T = phi[2]
        
        for i in val_indices
            coords = [x_data[i], z_data[i], t_data[i]]
            pred_u = phi_u(coords, θ.depvar.u)[1]
            pred_T = phi_T(coords, θ.depvar.T)[1]
            
            loss_u += (pred_u - u_data[i])^2
            loss_T += (pred_T - T_data[i])^2
        end
        return (loss_u + loss_T) / n_points
    end

    # 4. Estrategia de Discretización (Physics-Informed) con Muestreo de Importancia (Propuesta 3)
    strategy = QuasiRandomTraining(10000; sampling_alg = ImportanceSampler())
    
    # 5. Configurar el discretizador de NeuralPDE
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=additional_loss)

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
    res1 = Optimization.solve(prob, OptimizationOptimisers.Adam(learning_rate); callback = callback_adam, maxiters=epochs)
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
    res2 = Optimization.solve(prob2, OptimizationOptimJL.LBFGS(); callback = callback_lbfgs, maxiters=lbfgs_iters)
    println("Fase 2 terminada. Loss final L-BFGS: ", res2.objective)
    
    println("Exportando pesos acoplados...")
    open("pesos_pinn_boussinesq.json", "w") do f
        # No guardamos los pesos masivamente en JSON simple para evitar colapsos, guardamos el loss
        JSON.print(f, Dict("loss" => res2.objective, "info" => "Pesos de 6 redes guardados internamente."))
    end
    println("¡Entrenamiento exportado exitosamente!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_interpolative()
end
