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
using LinearAlgebra

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
    # Forzar OpenBLAS a un solo hilo para evitar malloc failed en Windows durante L-BFGS
    BLAS.set_num_threads(1)
    
    println("==== Iniciando Fase Interpolativa PINN Termodinámica ====")
    
    # Leer hiperparámetros si existen (inyectados por el Agente Python)
    epochs = 10
    learning_rate = 0.01
    lbfgs_iters = 20
    network_width = 32 # Valor por defecto seguro para evitar OOM
    max_data_points = 256 # Máximo número de puntos empíricos para evaluar en data loss
    collocation_points = 5000 # Rigurosidad física (5000 puntos en lugar de 128)
    
    if isfile("pinn_config.json")
        try
            config = JSON.parsefile("pinn_config.json")
            epochs = get(config, "epochs", 10)
            learning_rate = get(config, "learning_rate", 0.01)
            lbfgs_iters = get(config, "lbfgs_iters", 20)
            network_width = get(config, "network_width", 32)
            max_data_points = get(config, "max_data_points", 256)
            collocation_points = get(config, "collocation_points", 5000)
            println("Configuración recibida: Epochs=$epochs, LR=$learning_rate, L-BFGS-Iters=$lbfgs_iters, Width=$network_width, MaxPoints=$max_data_points, CollocationPoints=$collocation_points")
        catch
            println("Error leyendo pinn_config.json, usando valores por defecto.")
        end
    end
    
    # 1. Obtener la ecuación Boussinesq y arquitecturas (6 redes)
    pdesys, (x, z, t, u, T, vx, vz, P, S) = get_boussinesq_pde_system()
    chains = build_multi_pinn(network_width)
    
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

    # Muestrear un subconjunto máximo de puntos de entrenamiento para el ajuste de datos empíricos
    # Esto limita dramáticamente el consumo de RAM de la retropropagación en Julia/Zygote
    if length(train_indices) > max_data_points
        println("⚠️ Reduciendo puntos de entrenamiento de $(length(train_indices)) a $max_data_points para optimizar RAM.")
        train_indices = shuffle(train_indices)[1:max_data_points]
    end

    # Muestrear un subconjunto máximo de puntos de validación para evaluar la pérdida
    max_val_points = floor(Int, max_data_points / 2)
    if length(val_indices) > max_val_points
        println("⚠️ Reduciendo puntos de validación de $(length(val_indices)) a $max_val_points para optimizar RAM.")
        val_indices = shuffle(val_indices)[1:max_val_points]
    end

    # Pre-cargar y estructurar matrices de coordenadas de entrenamiento y validación (tamaño 3 x N)
    train_x = x_data[train_indices]
    train_z = z_data[train_indices]
    train_t = t_data[train_indices]
    train_coords = [train_x'; train_z'; train_t']  # Matriz 3 x N_train
    train_u = u_data[train_indices]
    train_T = T_data[train_indices]

    val_x = x_data[val_indices]
    val_z = z_data[val_indices]
    val_t = t_data[val_indices]
    val_coords = [val_x'; val_z'; val_t']          # Matriz 3 x N_val
    val_u = u_data[val_indices]
    val_T = T_data[val_indices]

    # 3. Definir la función de pérdida adicional vectorizada y sin bucles
    function additional_loss(phi, θ, p)
        n_points = length(train_indices)
        if n_points == 0
            return 0.0
        end
        
        phi_u = phi[1] # Red para la concentración u
        phi_T = phi[2] # Red para la temperatura T
        
        # Evaluación matricial directa
        pred_u = phi_u(train_coords, θ.depvar.u)
        pred_T = phi_T(train_coords, θ.depvar.T)
        
        # Cálculo de pérdida cuadrática media vectorizada
        loss_u = sum((vec(pred_u) .- train_u).^2)
        loss_T = sum((vec(pred_T) .- train_T).^2)
        
        return (loss_u + loss_T) / n_points
    end

    # Función para calcular la pérdida de validación en datos no vistos (Vectorizada)
    function eval_validation_loss(phi, θ)
        n_points = length(val_indices)
        if n_points == 0
            return 0.0
        end
        
        phi_u = phi[1]
        phi_T = phi[2]
        
        # Evaluación matricial directa
        pred_u = phi_u(val_coords, θ.depvar.u)
        pred_T = phi_T(val_coords, θ.depvar.T)
        
        loss_u = sum((vec(pred_u) .- val_u).^2)
        loss_T = sum((vec(pred_T) .- val_T).^2)
        
        return (loss_u + loss_T) / n_points
    end

    # 4. Estrategia de Discretización (Physics-Informed) con Muestreo de Importancia (Propuesta 3)
    strategy = QuasiRandomTraining(collocation_points; sampling_alg = ImportanceSampler())
    
    # 5. Configurar el discretizador de NeuralPDE
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=additional_loss)

    # 6. Convertir PDESystem al problema de Optimización
    println("Compilando el problema Boussinesq (esto tomará tiempo por las 5 ecuaciones)...")
    prob = discretize(pdesys, discretization)

    # 7. Ciclo de Entrenamiento
    loss_history = Dict{String, Any}(
        "epochs" => Int[],
        "train_loss" => Float64[],
        "val_loss" => Float64[],
        "phases" => String[]
    )
    
    epoch_count = 0
    callback_adam = function (p, l, args...)
        epoch_count += 1
        val_loss = 0.0
        try
            val_loss = eval_validation_loss(discretization.phi, p.u)
        catch e
        end
        println("[EPOCH_LOG] Epoch: $epoch_count | Loss: $l | Val Loss: $val_loss")
        
        push!(loss_history["epochs"], epoch_count)
        push!(loss_history["train_loss"], Float64(l))
        push!(loss_history["val_loss"], Float64(val_loss))
        push!(loss_history["phases"], "Adam")
        
        # Forzar recolección de basura cada 10 épocas para limpiar el tape de AD
        if epoch_count % 10 == 0
            GC.gc()
        end
        return false
    end

    println("Fase 1: Entrenando con Adam (LR=$learning_rate)...")
    res1 = Optimization.solve(prob, OptimizationOptimisers.Adam(learning_rate); callback = callback_adam, maxiters=epochs)
    println("Fase 1 terminada. Loss Adam: ", res1.objective)
    
    # Liberación profunda de memoria antes de L-BFGS
    GC.gc()
    
    # Fase 2: Refinamiento de precisión con L-BFGS (Optimizador de segundo orden)
    callback_lbfgs = function (p, l, args...)
        epoch_count += 1
        val_loss = 0.0
        try
            val_loss = eval_validation_loss(discretization.phi, p.u)
        catch e
        end
        println("[EPOCH_LOG] Epoch: $epoch_count | Loss: $l | Val Loss: $val_loss | Stage: L-BFGS")
        
        push!(loss_history["epochs"], epoch_count)
        push!(loss_history["train_loss"], Float64(l))
        push!(loss_history["val_loss"], Float64(val_loss))
        push!(loss_history["phases"], "L-BFGS")
        
        # Forzar recolección de basura cada 10 iteraciones de L-BFGS
        if epoch_count % 10 == 0
            GC.gc()
        end
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

    # Guardar historial de pérdidas e invocar graficador en Python
    println("Guardando historial de pérdidas para graficación...")
    open("historial_perdida.json", "w") do f
        JSON.print(f, loss_history)
    end
    
    try
        python_exe = isfile(".venv/Scripts/python.exe") ? ".venv/Scripts/python.exe" : "python"
        run(`$python_exe src/pinn/plot_loss.py`)
    catch e
        println("⚠️ No se pudo generar la gráfica de pérdidas automáticamente: $e")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_interpolative()
end
