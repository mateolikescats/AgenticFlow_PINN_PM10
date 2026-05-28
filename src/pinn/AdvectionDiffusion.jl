module AdvectionDiffusion

using ModelingToolkit
using Lux
using DomainSets
using NeuralPDE

export get_boussinesq_pde_system, build_multi_pinn, ImportanceSampler

"""
    get_boussinesq_pde_system()

Define el sistema de Ecuaciones Diferenciales Parciales acoplado para el Valle de Aburrá.
Modela un corte transversal vertical (x: ancho del valle, z: altura, t: tiempo).

Se integra la Aproximación de Boussinesq para modelar la Inversión Térmica:
1. Ecuación de Continuidad (Incompresibilidad)
2. Navier-Stokes 2D (Momentum X, Z con fuerza de flotabilidad por T)
3. Ecuación de Energía (Temperatura T)
4. Ecuación de Advección-Difusión-Reacción (Concentración u)
"""
function get_boussinesq_pde_system()
    @parameters x z t
    @variables u(..) T(..) vx(..) vz(..) P(..) S(..)
    
    Dt = Differential(t)
    Dx = Differential(x)
    Dz = Differential(z)
    Dxx = Differential(x)^2
    Dzz = Differential(z)^2

    # Parámetros termodinámicos y físicos adimensionalizados (basales)
    nu = 0.01      # Viscosidad cinemática
    alpha_T = 0.01 # Difusividad térmica
    D = 0.01       # Difusividad del PM2.5
    beta_g = 1.0   # Gravedad * Coeficiente de expansión térmica (Adimensional)
    T_ref = 0.0    # Temperatura de referencia (ej. tope de la capa límite)
    
    # Máscara Brinkman de relieve (Tazón del Valle de Aburrá)
    # h(x) = 0.4 * x^2, con suavizado delta = 0.05
    chi = 1.0 / (1.0 + exp(-((0.4 * x^2 - z) / 0.05)))
    
    # 1. Conservación de Masa (Incompresible)
    eq_mass = Dx(vx(x,z,t)) + Dz(vz(x,z,t)) ~ 0.0
    
    # 2. Momentum en X (Canalización transversal con penalización Brinkman)
    eq_mom_x = Dt(vx(x,z,t)) + vx(x,z,t)*Dx(vx(x,z,t)) + vz(x,z,t)*Dz(vx(x,z,t)) ~ 
               -Dx(P(x,z,t)) + nu * (Dxx(vx(x,z,t)) + Dzz(vx(x,z,t))) - (chi / 2e-3) * vx(x,z,t)
               
    # 3. Momentum en Z (Estratificación, aproximación de Boussinesq y penalización Brinkman)
    eq_mom_z = Dt(vz(x,z,t)) + vx(x,z,t)*Dx(vz(x,z,t)) + vz(x,z,t)*Dz(vz(x,z,t)) ~ 
               -Dz(P(x,z,t)) + nu * (Dxx(vz(x,z,t)) + Dzz(vz(x,z,t))) + beta_g * (T(x,z,t) - T_ref) - (chi / 2e-3) * vz(x,z,t)
               
    # 4. Transporte de Calor (Radiación / Termodinámica)
    eq_energy = Dt(T(x,z,t)) + vx(x,z,t)*Dx(T(x,z,t)) + vz(x,z,t)*Dz(T(x,z,t)) ~ 
                alpha_T * (Dxx(T(x,z,t)) + Dzz(T(x,z,t)))
                
    # 5. Advección-Difusión de Contaminantes (PM2.5 / PM10 con penalización Brinkman en subsuelo)
    eq_transport = Dt(u(x,z,t)) + vx(x,z,t)*Dx(u(x,z,t)) + vz(x,z,t)*Dz(u(x,z,t)) ~ 
                   D * (Dxx(u(x,z,t)) + Dzz(u(x,z,t))) + S(x,z,t) - (chi / 2e-3) * u(x,z,t)
    
    eqs = [eq_mass, eq_mom_x, eq_mom_z, eq_energy, eq_transport]
    
    # Dominio adimensional del corte transversal del valle
    # x en [-1, 1] son las laderas, z en [0, 1] es la altitud de la capa límite.
    domains = [
        x ∈ DomainSets.ClosedInterval(-1.0, 1.0),
        z ∈ DomainSets.ClosedInterval(0.0, 1.0),
        t ∈ DomainSets.ClosedInterval(0.0, 1.0)
    ]
    
    # Condiciones de frontera topológicas (Hard constraints)
    bcs = [
        # No-slip en el fondo y paredes del valle (vientos cero en la topografía)
        vx(-1.0, z, t) ~ 0.0, vx(1.0, z, t) ~ 0.0, vx(x, 0.0, t) ~ 0.0,
        vz(-1.0, z, t) ~ 0.0, vz(1.0, z, t) ~ 0.0, vz(x, 0.0, t) ~ 0.0,
        
        # Radiación/Inversión: Temperatura fría abajo (z=0), caliente arriba (z=1) simulando inversión
        T(x, 0.0, t) ~ -1.0, # Frío en el fondo
        T(x, 1.0, t) ~ 1.0,  # Capa de aire cálido arriba
        
        # Concentración de contaminantes (sin escape por el suelo o paredes)
        u(-1.0, z, t) ~ 0.0, u(1.0, z, t) ~ 0.0,
        
        # Condiciones Iniciales t=0 (estado base en reposo)
        vx(x, z, 0.0) ~ 0.0, vz(x, z, 0.0) ~ 0.0,
        P(x, z, 0.0)  ~ 0.0,
        T(x, z, 0.0)  ~ z,    # Gradiente inicial
        u(x, z, 0.0)  ~ 0.0
    ]
    
    @named pdesys = PDESystem(eqs, bcs, domains, [x,z,t], [u(x,z,t), T(x,z,t), vx(x,z,t), vz(x,z,t), P(x,z,t), S(x,z,t)])
    return pdesys, (x, z, t, u, T, vx, vz, P, S)
end

"""
    build_multi_pinn()

Construye 5 redes neuronales independientes (o múltiples cabezas) para aproximar
las variables dependientes [u, T, vx, vz, P].
"""
function build_multi_pinn()
    # Redes más anchas (64 neuronas) para modelar adecuadamente las 5 PDEs acopladas y el relieve
    make_net = () -> Lux.Chain(
        Lux.Dense(3, 64, Lux.tanh),
        Lux.Dense(64, 64, Lux.tanh),
        Lux.Dense(64, 64, Lux.tanh),
        Lux.Dense(64, 1)
    )
    
    # Red de emisión S con activación final softplus para garantizar S >= 0.0 (físicamente consistente)
    net_s = Lux.Chain(
        Lux.Dense(3, 64, Lux.tanh),
        Lux.Dense(64, 64, Lux.tanh),
        Lux.Dense(64, 64, Lux.tanh),
        Lux.Dense(64, 1, Lux.softplus)
    )
    
    return [make_net(), make_net(), make_net(), make_net(), make_net(), net_s]
end

# ==============================================================================
# ESTRATEGIA DE MUESTREO DE IMPORTANCIA ESPACIAL (Propuesta 3)
# ==============================================================================
const QMC = NeuralPDE.QuasiMonteCarlo

# Struct que hereda de la clase base de QuasiMonteCarlo
struct ImportanceSampler <: QMC.SamplingAlgorithm end

# Implementación de 4 argumentos: sample(n, d, S, T)
function QMC.sample(n::Integer, d::Integer, S::ImportanceSampler, T::Type)
    base_samples = QMC.sample(n, d, QMC.LatinHypercubeSample(), T)
    if d == 3 # Espacio-tiempo de las PDEs: [x, z, t]
        for i in 1:n
            # 1. Recuperar coordenada x en espacio físico [-1, 1] para calcular relieve
            x_phys = 2.0 * base_samples[1, i] - 1.0
            h_x = 0.4 * x_phys^2
            
            # 2. Aplicar muestreo de importancia bimodal vertical
            z_uni = base_samples[2, i]
            if rand() < 0.5
                # Sesgo hacia el suelo (z ≈ 0)
                z_new = z_uni^3.0
            else
                # Sesgo hacia la inversión térmica (z ≈ 0.5)
                z_new = 0.5 + 4.0 * (z_uni - 0.5)^3.0
            end
            
            # 3. Enmascaramiento de colocación (Estrategia 3):
            # Si el punto cae en el subsuelo profundo (z < h(x) - 0.05),
            # lo proyectamos a la zona activa [h(x) - 0.05, 1.0]
            cutoff = h_x - 0.05
            if z_new < cutoff
                z_new = cutoff + z_uni * (1.0 - cutoff)
            end
            
            base_samples[2, i] = clamp(z_new, 0.0, 1.0)
        end
    end
    return base_samples
end

# Implementación fallback de 3 argumentos: sample(n, d, S)
function QMC.sample(n::Integer, d::Integer, S::ImportanceSampler)
    return QMC.sample(n, d, S, Float64)
end

end # module
