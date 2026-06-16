using Pkg
Pkg.activate(".")
using JLD2, ComponentArrays, Statistics

function check_layers()
    checkpoint_path = joinpath(@__DIR__, "modelo_pinn_checkpoint.jld2")
    if !isfile(checkpoint_path)
        println("❌ Error: No se encontró el checkpoint.")
        return
    end
    
    @load checkpoint_path theta
    
    for var in [:u, :T, :vx, :vy, :vz, :P, :S]
        println("\n==================================")
        println("VARIABLE: ", var)
        println("==================================")
        
        net_theta = getproperty(theta.depvar, var)
        
        # Iterar sobre las capas del Lux Chain
        for layer_name in keys(net_theta)
            layer_params = getproperty(net_theta, layer_name)
            println("  Capa: ", layer_name)
            for param_name in keys(layer_params)
                val = getproperty(layer_params, param_name)
                println("    - ", param_name, ": size=", size(val), 
                        ", mean=", mean(val), 
                        ", std=", std(val), 
                        ", range=[", minimum(val), ", ", maximum(val), "]")
            end
        end
    end
end

check_layers()
