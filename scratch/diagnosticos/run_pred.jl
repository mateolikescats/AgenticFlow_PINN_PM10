include("../predict.jl")
run_prediction("input_points.json", "output_predictions.json", "scratch/modelo_pinn_checkpoint.jld2")
