using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
println("SUCCESS")

