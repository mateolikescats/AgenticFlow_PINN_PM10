using Pkg
Pkg.activate(".")
println("Instantiating project environment (this might download packages)...")
Pkg.instantiate()
println("Including AdvectionDiffusion.jl...")
include("src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
println("SUCCESS")
