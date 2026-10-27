include(joinpath(@__DIR__, "..", "inference", "predict.jl"))
using Statistics

# Métricas de ajuste de la PINN frente a la red oficial de SIATA, separadas por estaciones de
# entrenamiento y de validación (no vistas), con una línea base de interpolación IDW
# (distancia inversa al cuadrado) construida sólo con estaciones de entrenamiento.

function metricas(obs::Vector{Float64}, pred::Vector{Float64})
    e = pred .- obs
    return Dict(
        "n" => length(obs),
        "rmse" => sqrt(mean(e .^ 2)),
        "mae" => mean(abs.(e)),
        "sesgo" => mean(e),
        "r" => length(obs) > 2 && std(obs) > 0 && std(pred) > 0 ? cor(obs, pred) : NaN
    )
end

function idw(lat, lon, vecinos; p=2.0)
    num, den = 0.0, 0.0
    for (vlat, vlon, val) in vecinos
        d = hypot(lat - vlat, lon - vlon)
        w = 1.0 / d^p
        num += w * val
        den += w
    end
    return num / den
end

function evaluate_metrics(data_path::String="data/datos_siata_temporal.json",
                          model_path::String="models/modelo_pinn.jld2",
                          split_path::String="models/split_estaciones.json")
    config = JSON.parsefile("data/domain_config.json")
    conc_max = Float64(config["conc_max"])
    data = JSON.parsefile(data_path)
    split = JSON.parsefile(split_path)
    train_ids = Set(String.(split["train"]))
    val_ids = Set(String.(split["val"]))

    chains = build_multi_pinn()
    rng = Random.default_rng()
    phi = [LuxPhi(chains[i], Lux.setup(rng, chains[i])[2]) for i in 1:length(chains)]
    @load model_path theta

    pts = hcat([[Float64(d["x"]), Float64(d["y"]), Float64(d["z"]), Float64(d["t"])] for d in data]...)
    pinn = vec(clamp.(phi[1](pts, theta.depvar.u) .* conc_max, 0.0, Inf))

    # Agrupar por instante para la línea base IDW
    por_t = Dict{Float64, Vector{Int}}()
    for (i, d) in enumerate(data)
        push!(get!(por_t, Float64(d["t"]), Int[]), i)
    end

    registros = Dict{String, Any}[]
    for (i, d) in enumerate(data)
        id = string(d["id"])
        grupo = id in val_ids ? "validacion" : (id in train_ids ? "entrenamiento" : "otro")
        vecinos = [(Float64(data[j]["latitud"]), Float64(data[j]["longitud"]), Float64(data[j]["pm25"]))
                   for j in por_t[Float64(d["t"])] if string(data[j]["id"]) in train_ids && string(data[j]["id"]) != id]
        base = isempty(vecinos) ? NaN : idw(Float64(d["latitud"]), Float64(d["longitud"]), vecinos)
        push!(registros, Dict("id" => id, "grupo" => grupo, "t" => d["t"], "pm25" => Float64(d["pm25"]),
                              "pinn" => round(pinn[i], digits=3), "idw" => round(base, digits=3)))
    end

    resumen = Dict{String, Any}()
    for grupo in ["entrenamiento", "validacion"]
        rs = filter(r -> r["grupo"] == grupo && !isnan(r["idw"]), registros)
        obs = Float64[r["pm25"] for r in rs]
        resumen[grupo] = Dict(
            "pinn" => metricas(obs, Float64[r["pinn"] for r in rs]),
            "idw" => metricas(obs, Float64[r["idw"] for r in rs]),
            "estaciones" => length(unique(r["id"] for r in rs))
        )
    end
    por_estacion = Dict{String, Any}()
    for id in unique(r["id"] for r in registros)
        rs = filter(r -> r["id"] == id && !isnan(r["idw"]), registros)
        obs = Float64[r["pm25"] for r in rs]
        por_estacion[id] = Dict("grupo" => first(rs)["grupo"],
                                "pinn" => metricas(obs, Float64[r["pinn"] for r in rs]),
                                "idw" => metricas(obs, Float64[r["idw"] for r in rs]))
    end

    open("data/metricas_ajuste.json", "w") do f
        JSON.print(f, Dict("resumen" => resumen, "por_estacion" => por_estacion, "registros" => registros))
    end
    for grupo in ["entrenamiento", "validacion"]
        g = resumen[grupo]
        println("$grupo ($(g["estaciones"]) estaciones): PINN RMSE=$(round(g["pinn"]["rmse"], digits=2)) r=$(round(g["pinn"]["r"], digits=2)) | IDW RMSE=$(round(g["idw"]["rmse"], digits=2)) r=$(round(g["idw"]["r"], digits=2))")
    end
    println("✅ Métricas guardadas en data/metricas_ajuste.json")
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate_metrics()
end
