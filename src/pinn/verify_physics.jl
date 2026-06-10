using Pkg
Pkg.activate(".")
include("AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE, ModelingToolkit, JLD2, ComponentArrays, Lux, JSON

function verify_physics(model_path::String="modelo_pinn.jld2")
    println("==== Auditoría de Física Real (PVI y Divergencia) ====")
    
    if !isfile(model_path)
        println("❌ Error: No se encontró el archivo del modelo en $model_path")
        return
    end

    # 1. Recrear el sistema y la PINN
    pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S) = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())
    adaptive_strategy = GradientScaleAdaptiveLoss(100)
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=(phi, θ, p)->0.0, weight_strategy=adaptive_strategy)
    
    # 2. Cargar parámetros entrenados
    println("Cargando pesos del modelo...")
    @load model_path theta
    phi = discretization.phi

    # 3. Construir la grilla de evaluación 3D alineada con las predicciones
    t_val = 0.5
    if isfile("input_points.json")
        try
            inputs = JSON.parsefile("input_points.json")
            if !isempty(inputs)
                t_val = maximum([Float64(d["timestamp"]) for d in inputs])
                println("Alineando auditoría física con el t de las últimas predicciones (t_val = $t_val)")
            end
        catch e
            println("No se pudo leer input_points.json, usando t_val = 0.5 de fallback. Error: $e")
        end
    end

    println("Construyendo grilla 3D...")
    nx, ny, nz = 21, 21, 11
    x_vals = range(-1.0, 1.0, length=nx)
    y_vals = range(-1.0, 1.0, length=ny)
    z_vals = range(0.0, 1.0, length=nz)

    # Almacenar puntos
    pts = []
    for xi in x_vals
        for yj in y_vals
            for zk in z_vals
                push!(pts, [xi, yj, zk, t_val])
            end
        end
    end
    N = length(pts)
    coords = hcat(pts...) # Matriz (4, N)

    # 4. Calcular derivadas mediante diferencias centrales vectorizadas
    h = 1e-4
    println("Calculando divergencia del viento mediante diferencias centrales vectorizadas (h=$h)...")
    
    # Crear coordenadas desplazadas
    coords_xp = copy(coords); coords_xp[1, :] .+= h
    coords_xm = copy(coords); coords_xm[1, :] .-= h
    
    coords_yp = copy(coords); coords_yp[2, :] .+= h
    coords_ym = copy(coords); coords_ym[2, :] .-= h
    
    coords_zp = copy(coords); coords_zp[3, :] .+= h
    coords_zm = copy(coords); coords_zm[3, :] .-= h

    # Evaluar velocidades
    vx_xp = phi[3](coords_xp, theta.depvar.vx)
    vx_xm = phi[3](coords_xm, theta.depvar.vx)
    
    vy_yp = phi[4](coords_yp, theta.depvar.vy)
    vy_ym = phi[4](coords_ym, theta.depvar.vy)
    
    vz_zp = phi[5](coords_zp, theta.depvar.vz)
    vz_zm = phi[5](coords_zm, theta.depvar.vz)

    # Evaluar campos en la coordenada base
    vx_base = phi[3](coords, theta.depvar.vx)
    vy_base = phi[4](coords, theta.depvar.vy)
    vz_base = phi[5](coords, theta.depvar.vz)
    u_base = phi[1](coords, theta.depvar.u)
    T_base = phi[2](coords, theta.depvar.T)
    S_base = phi[7](coords, theta.depvar.S)

    # Diferencias centrales
    dvx_dx = (vx_xp .- vx_xm) ./ (2h)
    dvy_dy = (vy_yp .- vy_ym) ./ (2h)
    dvz_dz = (vz_zp .- vz_zm) ./ (2h)

    # Divergencia del viento y PVI
    div_v = dvx_dx .+ dvy_dy .+ dvz_dz
    abs_div = abs.(div_v)
    
    pvi = sum(abs_div) / N
    max_div = maximum(abs_div)
    min_div = minimum(div_v)
    max_div_raw = maximum(div_v)

    println("=== Métricas de Auditoría Física ===")
    println("Physics Violation Index (PVI - Divergencia Media Absoluta): $pvi")
    println("Divergencia Máxima Absoluta: $max_div")
    println("Divergencia Rango: [$min_div, $max_div_raw]")

    # 5. Exportar datos a JSON para graficar en Python
    results = Dict(
        "pvi" => pvi,
        "max_div" => max_div,
        "nx" => nx,
        "ny" => ny,
        "nz" => nz,
        "x" => collect(x_vals),
        "y" => collect(y_vals),
        "z" => collect(z_vals),
        "t" => t_val,
        "divergence" => collect(div_v),
        "vx" => collect(vx_base),
        "vy" => collect(vy_base),
        "vz" => collect(vz_base),
        "u" => collect(u_base),
        "T" => collect(T_base),
        "S" => collect(S_base)
    )

    out_file = "scratch/pvi_data.json"
    open(out_file, "w") do f
        JSON.print(f, results)
    end
    println("✅ Datos de divergencia guardados en $out_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_physics()
end
