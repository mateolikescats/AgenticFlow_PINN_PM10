using Pkg
Pkg.activate(".")
using JLD2, ComponentArrays, Lux, JSON, Random

# Estructura ligera para evaluar redes Lux de forma óptima sin compilación simbólica
struct LuxPhi
    chain::Lux.Chain
    state::NamedTuple
end
(lp::LuxPhi)(pts, θ) = first(Lux.apply(lp.chain, pts, θ, lp.state))

# Arquitectura Lux copiada localmente para evitar cargar la pesada librería NeuralPDE
function build_multi_pinn()
    make_net = () -> Lux.Chain(
        Lux.Dense(4, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1)
    )
    net_s = Lux.Chain(
        Lux.Dense(4, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1, Lux.softplus)
    )
    return [make_net(), make_net(), make_net(), make_net(), make_net(), make_net(), net_s]
end

function verify_physics(model_path::String="models/modelo_pinn.jld2")
    println("==== Auditoría de Física Real (PVI y Divergencia) ====")
    
    if !isfile(model_path)
        println("❌ Error: No se encontró el archivo del modelo en $model_path")
        return
    end

    # 1. Recrear la estructura de la PINN de forma ligera sin compilación simbólica
    chains = build_multi_pinn()
    rng = Random.default_rng()
    phi = [LuxPhi(chains[i], Lux.setup(rng, chains[i])[2]) for i in 1:length(chains)]
    
    # 2. Cargar parámetros entrenados
    println("Cargando pesos del modelo...")
    @load model_path theta

    # 3. Construir la grilla de evaluación 3D alineada con las predicciones
    t_val = 0.5
    if isfile("data/input_points.json")
        try
            inputs = JSON.parsefile("data/input_points.json")
            if !isempty(inputs)
                t_val = maximum([Float64(d["timestamp"]) for d in inputs])
                println("Alineando auditoría física con el t de las últimas predicciones (t_val = $t_val)")
            end
        catch e
            println("No se pudo leer data/input_points.json, usando t_val = 0.5 de fallback. Error: $e")
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

    # Sólo la atmósfera modelada: bajo el relieve parabólico (z < 0.4x² - 0.05) el ImportanceSampler
    # no coloca puntos, así que ahí la red no tiene restricción física y no debe entrar en la métrica.
    atm = vec(coords[3, :] .>= 0.4 .* coords[1, :].^2 .- 0.05)
    abs_div_atm = vec(abs_div)[atm]

    pvi = sum(abs_div_atm) / length(abs_div_atm)
    pvi_grilla_completa = sum(abs_div) / N
    max_div = maximum(abs_div_atm)
    min_div = minimum(vec(div_v)[atm])
    max_div_raw = maximum(vec(div_v)[atm])

    # Índice relativo: |div| / (|∂vx/∂x| + |∂vy/∂y| + |∂vz/∂z|) ∈ [0, 1], independiente de la magnitud del viento
    # (0 = los tres términos se compensan, 1 = ninguna compensación)
    suma_terminos = vec(abs.(dvx_dx) .+ abs.(dvy_dy) .+ abs.(dvz_dz))[atm]
    pvi_relativo = sum(abs_div_atm) / max(sum(suma_terminos), eps())

    # Divergencia en unidades físicas (1/s): longitudes del dominio en metros y velocidades en m/s
    config = JSON.parsefile("data/domain_config.json")
    lat_c = deg2rad((config["lat_min"] + config["lat_max"]) / 2)
    Lx = (config["lon_max"] - config["lon_min"]) * 111_320.0 * cos(lat_c)
    Ly = (config["lat_max"] - config["lat_min"]) * 110_574.0
    Lz = config["elev_max"] - config["elev_min"]
    w = config["wind_max"]
    div_fisica = vec(w .* (2 / Lx .* dvx_dx .+ 2 / Ly .* dvy_dy .+ 1 / Lz .* dvz_dz))[atm]
    div_fisica_media = sum(abs.(div_fisica)) / length(div_fisica)

    # NVR: fracción de la atmósfera con concentración negativa · TSI: fracción con ∂T/∂z > 0 en la capa de inversión
    T_zp = phi[2](coords_zp, theta.depvar.T)
    T_zm = phi[2](coords_zm, theta.depvar.T)
    dT_dz = vec((T_zp .- T_zm) ./ (2h))
    nvr = count(<(0), vec(u_base)[atm]) / count(atm)
    capa_inversion = atm .& (abs.(vec(coords[3, :]) .- 0.5) .<= 0.1)
    tsi_positivo = count(>(0), dT_dz[capa_inversion]) / max(count(capa_inversion), 1)

    println("=== Métricas de Auditoría Física (atmósfera modelada, $(count(atm)) de $N celdas) ===")
    println("Physics Violation Index (PVI - Divergencia Media Absoluta): $pvi")
    println("PVI sobre la grilla completa (incluye subsuelo, sólo referencia): $pvi_grilla_completa")
    println("PVI relativo (0 = masa conservada, 1 = sin compensación): $pvi_relativo")
    println("Divergencia física media |∇·v|: $div_fisica_media 1/s")
    println("Divergencia Máxima Absoluta: $max_div")
    println("Divergencia Rango: [$min_div, $max_div_raw]")
    println("NVR (fracción con u < 0): $nvr")
    println("TSI (fracción con ∂T/∂z > 0 en z ∈ [0.4, 0.6]): $tsi_positivo")

    # 5. Exportar datos a JSON para graficar en Python
    results = Dict(
        "pvi" => pvi,
        "pvi_grilla_completa" => pvi_grilla_completa,
        "pvi_relativo" => pvi_relativo,
        "div_fisica_media_1_s" => div_fisica_media,
        "nvr" => nvr,
        "tsi_positivo" => tsi_positivo,
        "atmosfera" => collect(atm),
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

    out_file = "data/pvi_data.json"
    open(out_file, "w") do f
        JSON.print(f, results)
    end
    println("✅ Datos de divergencia guardados en $out_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_physics()
end
