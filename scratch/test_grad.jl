using Pkg
Pkg.activate(".")
include("../src/pinn/AdvectionDiffusion.jl")
using .AdvectionDiffusion
using NeuralPDE
using ModelingToolkit
using Lux
using Optimization
using OptimizationOptimisers
using ComponentArrays

println("Preparing system...")
pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S) = get_boussinesq_pde_system()
chains = build_multi_pinn()

# 10 dummy data points
train_coords = rand(Float64, 4, 10)
u_data_vec = rand(Float64, 1, 10)
T_data_vec = rand(Float64, 1, 10)
meteo_coords = rand(Float64, 4, 5)
vx_meteo_vec = rand(Float64, 1, 5)
vy_meteo_vec = rand(Float64, 1, 5)

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

strategy = QuasiRandomTraining(16; sampling_alg=ImportanceSampler())
adaptive_strategy = GradientScaleAdaptiveLoss(10)

discretization = PhysicsInformedNN(chains, strategy;
    additional_loss=additional_loss,
    weight_strategy=adaptive_strategy)

println("Discretizing...")
prob = discretize(pdesys, discretization)

println("Computing initial loss...")
l0 = prob.f(prob.u0, prob.p)
println("Initial loss: ", l0)

println("Running one optimization step with Adam...")
opt = OptimizationOptimisers.Adam(0.01)
res = Optimization.solve(prob, opt, maxiters=2)
l1 = prob.f(res.u, prob.p)
println("Loss after 2 iterations of Adam: ", l1)

if l1 < l0
    println("SUCCESS: Loss decreased from ", l0, " to ", l1)
else
    println("WARNING: Loss did not decrease: l0=", l0, ", l1=", l1)
end
