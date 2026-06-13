using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using Lux, JLD2, ComponentArrays, Random, Statistics

function test_layers()
    checkpoint_path = joinpath(@__DIR__, "modelo_pinn_checkpoint.jld2")
    if !isfile(checkpoint_path)
        println("❌ Error: No se encontró el checkpoint.")
        return
    end
    
    @load checkpoint_path theta
    chains = build_multi_pinn()
    chain_u = chains[1]
    
    rng = Random.default_rng()
    _, st = Lux.setup(rng, chain_u)
    ps = theta.depvar.u
    
    # Evaluar en dos puntos extremos del espacio x: -1 y 1
    pt_neg = [ -1.0; 0.0; 0.2; 0.5 ]''
    pt_pos = [  1.0; 0.0; 0.2; 0.5 ]''
    
    y1_neg, _ = chain_u.layers.layer_1(pt_neg, ps.layer_1, st.layer_1)
    y1_pos, _ = chain_u.layers.layer_1(pt_pos, ps.layer_1, st.layer_1)
    
    y2_neg, _ = chain_u.layers.layer_2(y1_neg, ps.layer_2, st.layer_2)
    y2_pos, _ = chain_u.layers.layer_2(y1_pos, ps.layer_2, st.layer_2)
    
    y3_neg, _ = chain_u.layers.layer_3(y2_neg, ps.layer_3, st.layer_3)
    y3_pos, _ = chain_u.layers.layer_3(y2_pos, ps.layer_3, st.layer_3)
    
    y4_neg, _ = chain_u.layers.layer_4(y3_neg, ps.layer_4, st.layer_4)
    y4_pos, _ = chain_u.layers.layer_4(y3_pos, ps.layer_4, st.layer_4)
    
    println("Mean absolute differences in activations between x = -1.0 and x = 1.0:")
    println("  - Layer 1: ", mean(abs.(y1_neg .- y1_pos)))
    println("  - Layer 2: ", mean(abs.(y2_neg .- y2_pos)))
    println("  - Layer 3: ", mean(abs.(y3_neg .- y3_pos)))
    println("  - Layer 4 (Output): ", mean(abs.(y4_neg .- y4_pos)))
    
    println("\nDetailed Layer 4 output values:")
    println("  - Output at x = -1.0: ", y4_neg[1])
    println("  - Output at x =  1.0: ", y4_pos[1])
    println("  - Difference: ", abs(y4_neg[1] - y4_pos[1]))
end

test_layers()
