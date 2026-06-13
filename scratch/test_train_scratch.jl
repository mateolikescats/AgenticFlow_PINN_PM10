using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE, ModelingToolkit, JLD2, ComponentArrays, Lux, JSON, Statistics, Random
using Optimization, OptimizationOptimisers

function test_train_scratch()
    println("==== Test Training from Scratch ====")
    
    # 1. Obtener la ecuación Boussinesq y chains
    pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S) = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    
    # 2. Cargar datos
    pm_path = "datos_oficiales_pm25.json"
    wind_path = "datos_meteorologicos_viento.json"
    
    # Copia del cargador de datos de train_interpolative.jl
    # (Para no duplicar mucho, lo cargamos directamente)
    pm_data = JSON.parsefile(pm_path)
    wind_data = JSON.parsefile(wind_path)
    
    # Limitar datos para entrenamiento rápido de prueba (100 puntos)
    Random.seed!(42)
    train_data = shuffle(pm_data)[1:100]
    
    # Coordenadas y variables
    lat_min, lat_max = 6.0, 6.45
    lon_min, lon_max = -75.7, -75.3
    elev_min, elev_max = 1400.0, 3000.0
    
    x_data = Float64[]
    y_data = Float64[]
    z_data = Float64[]
    t_data = Float64[]
    u_data = Float64[]
    T_data = Float64[]
    
    # Construir coords de estaciones de viento
    station_coords = Dict{Int, Dict{String, Float64}}()
    for d in wind_data
        id_str = d["id"]
        clean_str = replace(id_str, "W-" => "")
        id_clean = round(Int, parse(Float64, clean_str))
        if !haskey(station_coords, id_clean)
            station_coords[id_clean] = Dict(
                "x" => Float64(d["x"]),
                "y" => Float64(d["y"]),
                "z" => Float64(d["z"]),
                "elevacion_real" => Float64(d["elevacion_real"])
            )
        end
    end
    
    for d in train_data
        id = round(Int, d["id"])
        if haskey(station_coords, id)
            coords = station_coords[id]
            push!(x_data, coords["x"])
            push!(y_data, coords["y"])
            push!(z_data, coords["z"])
            push!(t_data, 0.5) # tiempo fijo para test
            push!(u_data, clamp(d["pm25"] / 100.0, 0.0, 1.0))
            push!(T_data, 2.0 * coords["z"] - 1.0)
        end
    end
    
    if length(u_data) == 0
        println("❌ Error: No se alinearon datos.")
        return
    end
    
    train_coords = hcat([[x_data[i], y_data[i], z_data[i], t_data[i]] for i in 1:length(x_data)]...)
    u_data_vec = reshape(u_data, 1, :)
    T_data_vec = reshape(T_data, 1, :)
    
    # Loss adicional de datos
    additional_loss = (phi, θ, p) -> begin
        phi_u = phi[1]
        phi_T = phi[2]
        pred_u = phi_u(train_coords, θ.depvar.u)
        pred_T = phi_T(train_coords, θ.depvar.T)
        loss_u = sum((pred_u .- u_data_vec).^2) / length(u_data)
        loss_T = sum((pred_T .- T_data_vec).^2) / length(T_data)
        return loss_u + loss_T
    end
    
    strategy = QuasiRandomTraining(32)
    # Probar con y sin ponderación adaptativa
    discretization = PhysicsInformedNN(chains, strategy; 
        additional_loss=additional_loss,
        weight_strategy=GradientScaleAdaptiveLoss(100))
        
    prob = discretize(pdesys, discretization)
    
    # Entrenar por 200 epochs
    println("Entrenando por 200 épocas de Adam...")
    res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters=200)
    theta = res.u
    
    # Evaluar en un rango de x
    phi = discretization.phi
    x_vals = -1.0:0.5:1.0
    N_test = length(x_vals)
    test_pts = Matrix{Float64}(undef, 4, N_test)
    for i in 1:N_test
        test_pts[1, i] = x_vals[i]
        test_pts[2, i] = 0.0
        test_pts[3, i] = 0.2
        test_pts[4, i] = 0.5
    end
    
    u_pred = phi[1](test_pts, theta.depvar.u)
    T_pred = phi[2](test_pts, theta.depvar.T)
    
    println("\n=== Resultados post-entrenamiento rápido ===")
    for i in 1:N_test
        println("x = ", round(x_vals[i], digits=2), 
                " | u_pred = ", round(u_pred[i], digits=6), 
                " | T_pred = ", round(T_pred[i], digits=6))
    end
    
    # Guardar pesos de prueba para ver si cambiaron
    w1_u = theta.depvar.u.layer_1.weight
    println("\nStats de layer_1 de u:")
    println("  - Mean: ", mean(w1_u))
    println("  - StdDev: ", std(w1_u))
end

test_train_scratch()
