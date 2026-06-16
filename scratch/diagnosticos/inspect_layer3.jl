using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using Lux, JLD2, ComponentArrays, Random, Statistics

function inspect_l3()
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
    
    pt_neg = [ -1.0; 0.0; 0.2; 0.5 ]''
    pt_pos = [  1.0; 0.0; 0.2; 0.5 ]''
    
    y1_neg, _ = chain_u.layers.layer_1(pt_neg, ps.layer_1, st.layer_1)
    y1_pos, _ = chain_u.layers.layer_1(pt_pos, ps.layer_1, st.layer_1)
    
    y2_neg, _ = chain_u.layers.layer_2(y1_neg, ps.layer_2, st.layer_2)
    y2_pos, _ = chain_u.layers.layer_2(y1_pos, ps.layer_2, st.layer_2)
    
    # Evaluar pre-activaciones de layer_3
    # En Lux, una capa Dense calcula: activation.(weight * x .+ bias)
    # y2_neg es de tamaño (32, 1)
    W3 = ps.layer_3.weight
    b3 = ps.layer_3.bias
    
    pre3_neg = W3 * y2_neg .+ b3
    pre3_pos = W3 * y2_pos .+ b3
    
    println("Stats of y2_neg:")
    println("  - Mean: ", mean(y2_neg), ", Std: ", std(y2_neg))
    println("Stats of y2_pos:")
    println("  - Mean: ", mean(y2_pos), ", Std: ", std(y2_pos))
    println("Stats of difference (y2_neg - y2_pos):")
    diff_y2 = y2_neg .- y2_pos
    println("  - Mean abs: ", mean(abs.(diff_y2)), ", Std: ", std(diff_y2))
    println("  - Sample diff values (first 5): ", diff_y2[1:5])
    
    println("\nStats of W3:")
    println("  - Size: ", size(W3), ", Mean: ", mean(W3), ", Std: ", std(W3))
    
    println("\nStats of pre3_neg (W3 * y2 + b3):")
    println("  - Mean: ", mean(pre3_neg), ", Std: ", std(pre3_neg))
    println("Stats of difference (pre3_neg - pre3_pos):")
    diff_pre3 = pre3_neg .- pre3_pos
    println("  - Mean abs: ", mean(abs.(diff_pre3)), ", Std: ", std(diff_pre3))
    println("  - Sample diff values (first 5): ", diff_pre3[1:5])
end

inspect_l3()
