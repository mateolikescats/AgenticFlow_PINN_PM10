using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE, ModelingToolkit, JLD2, ComponentArrays, Lux, JSON

function test_grid()
    checkpoint_path = joinpath(@__DIR__, "modelo_pinn_checkpoint.jld2")
    if !isfile(checkpoint_path)
        println("❌ Error: No se encontró el checkpoint.")
        return
    end
    
    # 1. Crear red y cargar pesos
    pdesys, _ = get_boussinesq_pde_system()
    chains = build_multi_pinn()
    strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())
    adaptive_strategy = GradientScaleAdaptiveLoss(100)
    discretization = PhysicsInformedNN(chains, strategy; additional_loss=(phi, θ, p)->0.0, weight_strategy=adaptive_strategy)
    
    @load checkpoint_path theta
    phi = discretization.phi
    
    # 2. Probar cuadrícula en z de 0 a 1 (con x=0, y=0, t=0.5)
    z_vals = 0.0:0.1:1.0
    N = length(z_vals)
    pts = Matrix{Float64}(undef, 4, N)
    for i in 1:N
        pts[1, i] = 0.0
        pts[2, i] = 0.0
        pts[3, i] = z_vals[i]
        pts[4, i] = 0.5
    end
    
    u_pred = phi[1](pts, theta.depvar.u)
    T_pred = phi[2](pts, theta.depvar.T)
    vx_pred = phi[3](pts, theta.depvar.vx)
    vy_pred = phi[4](pts, theta.depvar.vy)
    S_pred = phi[7](pts, theta.depvar.S)
    
    println("Outputs for varying z (from 0.0 to 1.0):")
    for i in 1:N
        println("z = ", round(z_vals[i], digits=2), 
                " | u = ", round(u_pred[i], digits=6), 
                " | T = ", round(T_pred[i], digits=6),
                " | vx = ", round(vx_pred[i], digits=6),
                " | vy = ", round(vy_pred[i], digits=6),
                " | S = ", round(S_pred[i], digits=6))
    end
end

test_grid()
