using Pkg
Pkg.activate(".")
using JLD2, ComponentArrays, Statistics

# Load weights from the root folder
model_path = joinpath(@__DIR__, "../modelo_pinn.jld2")
@load model_path theta

println("=== Estadísticas de los parámetros del modelo ===")
for var in [:u, :T, :vx, :vy, :vz, :P, :S]
    w = getproperty(theta.depvar, var)
    println(var, " -> length: ", length(w), ", mean: ", mean(w), ", std: ", std(w), ", min: ", minimum(w), ", max: ", maximum(w))
end
