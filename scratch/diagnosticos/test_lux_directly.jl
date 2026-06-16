using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using Lux, JLD2, ComponentArrays, Random, Statistics

function test_lux_directly()
    checkpoint_path = joinpath(@__DIR__, "modelo_pinn_checkpoint.jld2")
    if !isfile(checkpoint_path)
        println("❌ Error: No se encontró el checkpoint.")
        return
    end
    
    @load checkpoint_path theta
    
    # 1. Recrear el Lux chain para u
    chains = build_multi_pinn()
    chain_u = chains[1]
    
    # 2. Inicializar estado st
    rng = Random.default_rng()
    _, st = Lux.setup(rng, chain_u)
    
    # 3. Obtener los parámetros ps desde theta.depvar.u
    ps = theta.depvar.u
    
    # 4. Probar cuadrícula en x de -1 a 1 (con y=0, z=0.2, t=0.5)
    x_vals = -1.0:0.2:1.0
    N = length(x_vals)
    pts = Matrix{Float64}(undef, 4, N)
    for i in 1:N
        pts[1, i] = x_vals[i]
        pts[2, i] = 0.0
        pts[3, i] = 0.2
        pts[4, i] = 0.5
    end
    
    # 5. Evaluar directamente con Lux
    y, _ = chain_u(pts, ps, st)
    
    println("Lux Direct Outputs for varying x (from -1.0 to 1.0):")
    for i in 1:N
        println("x = ", round(x_vals[i], digits=2), " | u_pred = ", y[1, i])
    end
end

test_lux_directly()
