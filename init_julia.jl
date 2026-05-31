using Pkg
Pkg.activate(".")
Pkg.add([
    "NeuralPDE", 
    "ModelingToolkit", 
    "Lux", 
    "Optimization", 
    "OptimizationOptimJL", 
    "OptimizationOptimisers", 
    "JSON", 
    "ComponentArrays"
])
Pkg.precompile()
