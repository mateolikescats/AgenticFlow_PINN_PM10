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
        println("⚠️ Archivo de datos no encontrado: $filepath. Usando datos ficticios.")
        # Generar datos ficticios (x, y, t, u)
        return [
            Dict("x" => 0.0, "y" => 0.0, "t" => 0.5, "u" => 0.5),
            Dict("x" => 0.5, "y" => 0.5, "t" => 0.8, "u" => 0.2)
        ]
    end
    return JSON.parsefile(filepath)
end

function train_interpolative(data_path="datos_siata.json", epochs=500)
    println("==== Iniciando Fase Interpolativa PINN ====")
    
    # 1. Obtener la ecuación y arquitectura
    pdesys, (x, y, t, u) = get_pde_system(0.01, 1.0)
    chain = build_pinn()
    
    # 2. Cargar datos empíricos (SIATA preprocesado)
    data = load_data(data_path)
    # Extraer arrays para evaluación rápida
    x_data = Float64[d["x"] for d in data]
    y_data = Float64[d["y"] for d in data]
    t_data = Float64[d["t"] for d in data]
    u_data = Float64[d["u"] for d in data]

    # 3. Definir la función de pérdida adicional (Ajuste a los datos de SIATA)
    # phi(x, θ) es la función de predicción de NeuralPDE
    # En NeuralPDE, x es una matriz donde cada columna es un punto (x, y, t)
    function additional_loss(phi, θ, p)
        loss = 0.0
        n_points = length(u_data)
        for i in 1:n_points
            # phi espera una matriz columna: [x, y, t]'
            coords = [x_data[i], y_data[i], t_data[i]]
            pred = phi(coords, θ)[1]
            loss += (pred - u_data[i])^2
        end
        return loss / max(1, n_points)
    end

    # 4. Estrategia de Discretización (Physics-Informed)
    # Usamos GridTraining o QuasiRandomTraining en el dominio
    strategy = QuasiRandomTraining(256)
    
    # 5. Configurar el discretizador de NeuralPDE
    discretization = PhysicsInformedNN(chain, strategy; additional_loss=additional_loss)

    # 6. Convertir PDESystem al problema de Optimización de SciML
    println("Compilando el problema de optimización (esto puede tardar unos segundos)...")
    prob = discretize(pdesys, discretization)

    # 7. Ciclo de Entrenamiento
    # Usamos Adam para los primeros epochs
    res = Optimization.solve(prob, Adam(0.01); maxiters=epochs)
    
    println("Fase interpolativa terminada. Loss final: ", res.objective)
    
    # Guardar pesos optimizados a JSON
    println("Exportando pesos pre-acondicionados...")
    # Convertir ComponentArray a Dict/Vector para JSON
    weights_vec = Float64.(res.u)
    open("pesos_pinn.json", "w") do f
        JSON.print(f, Dict("loss" => res.objective, "weights" => weights_vec))
    end
    println("¡Entrenamiento exportado exitosamente!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_interpolative()
end
