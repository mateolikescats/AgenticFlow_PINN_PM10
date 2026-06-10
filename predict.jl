using Pkg
Pkg.activate(".")
include("src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE, ModelingToolkit, JLD2, ComponentArrays, Lux, JSON

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

    # 7. Des-adimensionalizar y consolidar resultados
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
            "pred_emision_S_ug_m3_s" => round(emision_est, digits=6) # <-- Predicción de la PINN Inversa!
        ))
    end

    open(output_path, "w") do f
        JSON.print(f, output_data, 4)
    end
    println("✅ ¡Predicciones guardadas exitosamente en '$output_path'!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_prediction()
end
