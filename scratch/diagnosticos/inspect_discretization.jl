using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE
using ModelingToolkit
using Lux
using ComponentArrays
using JSON
using JLD2

# Load system
pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S) = get_boussinesq_pde_system()
chains = build_multi_pinn()

# Create dummy data coordinates for testing
train_coords = rand(4, 10)
u_data_vec = rand(1, 10)
T_data_vec = rand(1, 10)
meteo_coords = rand(4, 5)
vx_meteo_vec = rand(1, 5)
vy_meteo_vec = rand(1, 5)

additional_loss = (phi, θ, p) -> begin
    phi_u = phi[1]
    phi_T = phi[2]
    phi_vx = phi[3]
    phi_vy = phi[4]
    pred_u = phi_u(train_coords, θ.depvar.u)
    pred_T = phi_T(train_coords, θ.depvar.T)
    loss_u = sum((pred_u .- u_data_vec).^2) / 10
    loss_T = sum((pred_T .- T_data_vec).^2) / 10
    pred_vx = phi_vx(meteo_coords, θ.depvar.vx)
    pred_vy = phi_vy(meteo_coords, θ.depvar.vy)
    loss_v = (sum((pred_vx .- vx_meteo_vec).^2) + sum((pred_vy .- vy_meteo_vec).^2)) / 5
    return loss_u + loss_T + loss_v
end

strategy = QuasiRandomTraining(128; sampling_alg=ImportanceSampler())
adaptive_strategy = GradientScaleAdaptiveLoss(100)

discretization = PhysicsInformedNN(chains, strategy;
    additional_loss=additional_loss,
    weight_strategy=adaptive_strategy)

println("Running symbolic_discretize...")
sym_prob = NeuralPDE.symbolic_discretize(pdesys, discretization)

println("Running discretize...")
prob = discretize(pdesys, discretization)
u0 = prob.u0
println("Evaluating raw PDE loss functions with (u0)...")
try
    pde_losses = [f(u0) for f in sym_prob.loss_functions.pde_loss_functions]
    println("Raw PDE losses with (u0): ", pde_losses)
catch e
    println("Failed raw PDE with (u0): ", e)
end

println("Evaluating raw BC loss functions with (u0)...")
try
    bc_losses = [f(u0) for f in sym_prob.loss_functions.bc_loss_functions]
    println("Raw BC losses with (u0): ", bc_losses)
catch e
    println("Failed raw BC with (u0): ", e)
end

println("Evaluating adaptive PDE loss functions with (u0, prob.p)...")
try
    pde_losses_adap = [f(u0, prob.p) for f in sym_prob.pde_loss_functions]
    println("Adaptive PDE losses with (u0, prob.p): ", pde_losses_adap)
catch e
    println("Failed adaptive PDE with (u0, prob.p): ", e)
end



