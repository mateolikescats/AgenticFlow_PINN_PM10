using Pkg
Pkg.activate(".")
using JSON, Statistics

function describe_wind()
    data = JSON.parsefile("datos_meteorologicos_viento.json")
    xs = [Float64(d["x"]) for d in data]
    ys = [Float64(d["y"]) for d in data]
    zs = [Float64(d["z"]) for d in data]
    elevs = [Float64(d["elevacion_real"]) for d in data]
    
    println("Wind Coordinate Statistics:")
    println("  - Count: ", length(data))
    println("  - x: range = [", minimum(xs), ", ", maximum(xs), "], mean = ", mean(xs))
    println("  - y: range = [", minimum(ys), ", ", maximum(ys), "], mean = ", mean(ys))
    println("  - z: range = [", minimum(zs), ", ", maximum(zs), "], mean = ", mean(zs))
    println("  - elevacion_real: range = [", minimum(elevs), ", ", maximum(elevs), "], mean = ", mean(elevs))
end

describe_wind()
