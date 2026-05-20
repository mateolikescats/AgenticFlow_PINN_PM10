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
    if isfile("pinn_config.json")
        try
            config = JSON.parsefile("pinn_config.json")
            epochs = get(config, "epochs", 100)
            learning_rate = get(config, "learning_rate", 0.01)
            println("Configuración recibida del Agente: Epochs=$epochs, LR=$learning_rate")
        catch
            println("Error leyendo pinn_config.json, usando valores por defecto.")
        end
    end
    
    # 1. Obtener la ecuación Boussinesq y arquitecturas (5 redes)
    pdesys, (x, z, t, u, T, vx, vz, P) = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    
    # 2. Cargar datos empíricos
    data = load_data(data_path)
    x_data = Float64[d["x"] for d in data]
    z_data = Float64[d["z"] for d in data]
    t_data = Float64[d["t"] for d in data]
    u_data = Float64[d["u"] for d in data]
    T_data = Float64[d["T"] for d in data]

    # 3. Definir la función de pérdida adicional (Ajuste a los datos de SIATA)
    # phi es una tupla de funciones, una por cada red: [phi_u, phi_T, phi_vx, phi_vz, phi_P]
    # theta es una tupla/ComponentArray de los pesos de cada red
    function additional_loss(phi, θ, p)
        loss_u = 0.0
        loss_T = 0.0
        n_points = length(u_data)
        
        phi_u = phi[1] # Red para la concentración u
        phi_T = phi[2] # Red para la temperatura T
        
        for i in 1:n_points
            coords = [x_data[i], z_data[i], t_data[i]]
            pred_u = phi_u(coords, θ.depvar.u)[1]
            pred_T = phi_T(coords, θ.depvar.T)[1]
            
            loss_u += (pred_u - u_data[i])^2
            loss_T += (pred_T - T_data[i])^2
        end
        return (loss_u + loss_T) / max(1, n_points)
    end

    # 4. Estrategia de Discretización (Physics-Informed)
    strategy = QuasiRandomTraining(128) # Puntos de colocación reducidos por complejidad
    
    # 5. Configurar el discretizador de NeuralPDE
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=additional_loss)

    # 6. Convertir PDESystem al problema de Optimización
    println("Compilando el problema Boussinesq (esto tomará tiempo por las 5 ecuaciones)...")
    prob = discretize(pdesys, discretization)

    # 7. Ciclo de Entrenamiento
    println("Entrenando con Adam (LR=$learning_rate)...")
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(learning_rate); maxiters=epochs)
    
    println("Fase interpolativa terminada. Loss final: ", res.objective)
    
    println("Exportando pesos acoplados...")
    open("pesos_pinn_boussinesq.json", "w") do f
        # No guardamos los pesos masivamente en JSON simple para evitar colapsos, guardamos el loss
        JSON.print(f, Dict("loss" => res.objective, "info" => "Pesos de 5 redes guardados internamente."))
    end
    println("¡Entrenamiento exportado exitosamente!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_interpolative()
end
