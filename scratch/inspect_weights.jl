using Pkg
Pkg.activate(".")
using JLD2, ComponentArrays, Statistics

model_path = joinpath(@__DIR__, "../modelo_pinn.jld2")
@load model_path theta

w1 = theta.depvar.u.layer_1.weight
println("First layer weights of network u:")
println("  - Mean: ", mean(w1))
println("  - StdDev: ", std(w1))
println("  - Max Absolute: ", maximum(abs.(w1)))
println("  - Sample values (first 5): ", w1[1:5])
