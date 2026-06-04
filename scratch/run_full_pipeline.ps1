Write-Host "=== STARTING FULL AUTOMATED PINN PIPELINE ==="

# 1. Run Julia training
Write-Host "Step 1: Running Julia Training (20k Adam + 1500 L-BFGS)..."
julia --project=. src/pinn/train_interpolative.jl
if ($LASTEXITCODE -ne 0) {
    Write-Error "Julia training failed!"
    exit 1
}

# 2. Run physics verification
Write-Host "Step 2: Running Physics Verification (PVI calculation)..."
julia --project=. src/pinn/verify_physics.jl
if ($LASTEXITCODE -ne 0) {
    Write-Error "Physics verification failed!"
    exit 1
}

# 3. Generate plots
Write-Host "Step 3: Generating loss curves and PVI maps in Python..."
python scratch/plot_physics.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python plotting failed!"
    exit 1
}

# 4. Compile LaTeX Report
Write-Host "Step 4: Compiling LaTeX Technical Report..."
cd reporte
pdflatex -interaction=nonstopmode reporte_final_ejecucion.tex
pdflatex -interaction=nonstopmode reporte_final_ejecucion.tex
cd ..

# 5. Git commit and push
Write-Host "Step 5: Staging, committing, and pushing to origin/cristian..."
git add .gitignore Project.toml README.md pesos_pinn_boussinesq.json pinn_config.json modelo_pinn.jld2
git add reporte/reporte_final_ejecucion.pdf reporte/reporte_final_ejecucion.tex reporte/curvas_convergencia.png reporte/mapa_divergencia_pvi.png
git add src/pinn/train_interpolative.jl
git commit -m "merge: training on real data completed, PVI audited, and PDF report compiled"
git push origin cristian

Write-Host "=== PIPELINE COMPLETED SUCCESSFULLY! ==="
