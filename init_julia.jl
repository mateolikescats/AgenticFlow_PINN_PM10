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
    "ComponentArrays",
    "JLD2",
    "DomainSets"
])
Pkg.precompile()
