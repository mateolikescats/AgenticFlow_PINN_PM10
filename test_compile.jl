using Pkg
Pkg.add("DomainSets")
include("src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
println("SUCCESS")
