using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE, ModelingToolkit, JLD2, ComponentArrays, Lux, JSON, Statistics

function test_checkpoint_predictions()
    input_path = joinpath(@__DIR__, "../input_points.json")
    checkpoint_path = joinpath(@__DIR__, "modelo_pinn_checkpoint.jld2")
    
    if !isfile(checkpoint_path)
        println("❌ Error: No se encontró el checkpoint en $checkpoint_path")
        return
    end
    if !isfile(input_path)
        println("❌ Error: No se encontró input_points.json en $input_path")
        return
    end
    
    inputs = JSON.parsefile(input_path)
    N = length(inputs)
    
    # Adimensionalizar
    lat_min, lat_max = 6.0, 6.45
    lon_min, lon_max = -75.7, -75.3
    elev_min, elev_max = 1400.0, 3000.0
    
    pts = Matrix{Float64}(undef, 4, N)
    for i in 1:N
        d = inputs[i]
        lat = Float64(d["latitud"])
        lon = Float64(d["longitud"])
        elev = Float64(d["elevacion"])
        t_val = Float64(d["timestamp"])
        
        x_scaled = 2.0 * (lon - lon_min) / (lon_max - lon_min) - 1.0
        y_scaled = 2.0 * (lat - lat_min) / (lat_max - lat_min) - 1.0
        z_scaled = (elev - elev_min) / (elev_max - elev_min)
        
        pts[1, i] = x_scaled
        pts[2, i] = y_scaled
        pts[3, i] = z_scaled
        pts[4, i] = t_val
    end
    
    # Recrear PINN
    pdesys, _ = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())
    adaptive_strategy = GradientScaleAdaptiveLoss(100)
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=(phi, θ, p)->0.0, weight_strategy=adaptive_strategy)
    
    println("Cargando checkpoint desde $checkpoint_path...")
    @load checkpoint_path theta
    phi = discretization.phi
    
    # Evaluar
    u_pred = phi[1](pts, theta.depvar.u) # PM2.5 (adimensional)
    T_pred = phi[2](pts, theta.depvar.T) # Temp
    vx_pred = phi[3](pts, theta.depvar.vx)
    vy_pred = phi[4](pts, theta.depvar.vy)
    vz_pred = phi[5](pts, theta.depvar.vz)
    S_pred = phi[7](pts, theta.depvar.S)
    
    # Des-adimensionalizar
    pm25_est = clamp.(u_pred * 100.0, 0.0, Inf)
    vx_est = vx_pred * 10.0
    vy_est = vy_pred * 10.0
    vz_est = vz_pred * 10.0
    emision_est = S_pred * (100.0 / 3600.0)
    
    println("\n=== Estadísticas de predicción en los puntos de entrada ===")
    println("PM2.5 (ug/m3):")
    println("  - Mean: ", mean(pm25_est))
    println("  - StdDev: ", std(pm25_est))
    println("  - Range: [", minimum(pm25_est), ", ", maximum(pm25_est), "]")
    
    println("Viento VX (m/s):")
    println("  - Mean: ", mean(vx_est))
    println("  - StdDev: ", std(vx_est))
    println("  - Range: [", minimum(vx_est), ", ", maximum(vx_est), "]")
    
    println("Viento VY (m/s):")
    println("  - Mean: ", mean(vy_est))
    println("  - StdDev: ", std(vy_est))
    println("  - Range: [", minimum(vy_est), ", ", maximum(vy_est), "]")
    
    println("Viento VZ (m/s):")
    println("  - Mean: ", mean(vz_est))
    println("  - StdDev: ", std(vz_est))
    println("  - Range: [", minimum(vz_est), ", ", maximum(vz_est), "]")
    
    println("Emisión S (ug/m3/s):")
    println("  - Mean: ", mean(emision_est))
    println("  - StdDev: ", std(emision_est))
    println("  - Range: [", minimum(emision_est), ", ", maximum(emision_est), "]")
    
    # Mostrar las predicciones para las primeras 5 estaciones
    println("\nPrimeras 5 estaciones:")
    for i in 1:min(5, N)
        println("Estación $i (Lat: ", inputs[i]["latitud"], ", Lon: ", inputs[i]["longitud"], "):")
        println("  * PM2.5: ", round(pm25_est[i], digits=3), " ug/m3")
        println("  * Viento: [", round(vx_est[i], digits=3), ", ", round(vy_est[i], digits=3), ", ", round(vz_est[i], digits=3), "] m/s")
        println("  * Emisión (S): ", round(emision_est[i], digits=6), " ug/m3/s")
    end
end

test_checkpoint_predictions()
