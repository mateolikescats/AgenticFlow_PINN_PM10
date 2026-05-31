module AdvectionDiffusion

using ModelingToolkit
using Lux
using DomainSets
using NeuralPDE

export get_boussinesq_pde_system, build_multi_pinn, ImportanceSampler

"""
    get_boussinesq_pde_system()

Define el sistema de Ecuaciones Diferenciales Parciales 3D acoplado para el Valle de Aburrá.
Modela un volumen (x: ancho, y: largo, z: altura, t: tiempo).

Se integra la Aproximación de Boussinesq y Sedimentación:
1. Ecuación de Continuidad 3D (Incompresibilidad)
2. Navier-Stokes 3D (Momentum X, Y, Z con fuerza de flotabilidad por T)
3. Ecuación de Energía (Temperatura T)
4. Ecuación de Advección-Difusión-Reacción (Concentración u con velocidad de sedimentación)
"""
function get_boussinesq_pde_system()
    @parameters x y z t
    @variables u(..) T(..) vx(..) vy(..) vz(..) P(..) S(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2

    # Parámetros termodinámicos y físicos adimensionalizados (basales)
    nu = 0.01      # Viscosidad cinemática
    alpha_T = 0.01 # Difusividad térmica
    D = 0.01       # Difusividad del PM2.5
    beta_g = 1.0   # Gravedad * Coeficiente de expansión térmica (Adimensional)
    T_ref = 0.0    # Temperatura de referencia
    v_s = 0.05     # Velocidad de sedimentación del PM2.5 (gravedad)

    # 1. Conservación de Masa (Incompresible 3D)
    eq_mass = Dx(vx(x, y, z, t)) + Dy(vy(x, y, z, t)) + Dz(vz(x, y, z, t)) ~ 0.0

    # 2. Momentum en X (Ancho del valle)
    eq_mom_x = Dt(vx(x, y, z, t)) + vx(x, y, z, t) * Dx(vx(x, y, z, t)) + vy(x, y, z, t) * Dy(vx(x, y, z, t)) + vz(x, y, z, t) * Dz(vx(x, y, z, t)) ~
        -Dx(P(x, y, z, t)) + nu * (Dxx(vx(x, y, z, t)) + Dyy(vx(x, y, z, t)) + Dzz(vx(x, y, z, t)))

    # 3. Momentum en Y (Largo del valle - Eje longitudinal)
    eq_mom_y = Dt(vy(x, y, z, t)) + vx(x, y, z, t) * Dx(vy(x, y, z, t)) + vy(x, y, z, t) * Dy(vy(x, y, z, t)) + vz(x, y, z, t) * Dz(vy(x, y, z, t)) ~
        -Dy(P(x, y, z, t)) + nu * (Dxx(vy(x, y, z, t)) + Dyy(vy(x, y, z, t)) + Dzz(vy(x, y, z, t)))

    # 4. Momentum en Z (Estratificación vertical y Boussinesq)
    eq_mom_z = Dt(vz(x, y, z, t)) + vx(x, y, z, t) * Dx(vz(x, y, z, t)) + vy(x, y, z, t) * Dy(vz(x, y, z, t)) + vz(x, y, z, t) * Dz(vz(x, y, z, t)) ~
        -Dz(P(x, y, z, t)) + nu * (Dxx(vz(x, y, z, t)) + Dyy(vz(x, y, z, t)) + Dzz(vz(x, y, z, t))) + beta_g * (T(x, y, z, t) - T_ref)

    # 5. Transporte de Calor (Radiación / Termodinámica)
    eq_energy = Dt(T(x, y, z, t)) + vx(x, y, z, t) * Dx(T(x, y, z, t)) + vy(x, y, z, t) * Dy(T(x, y, z, t)) + vz(x, y, z, t) * Dz(T(x, y, z, t)) ~
        alpha_T * (Dxx(T(x, y, z, t)) + Dyy(T(x, y, z, t)) + Dzz(T(x, y, z, t)))

    # 6. Advección-Difusión de PM2.5 (Con Sedimentación v_s)
    # Se añade la caída gravitacional en el término z: (vz - v_s)
    eq_transport = Dt(u(x, y, z, t)) + vx(x, y, z, t) * Dx(u(x, y, z, t)) + vy(x, y, z, t) * Dy(u(x, y, z, t)) + (vz(x, y, z, t) - v_s) * Dz(u(x, y, z, t)) ~
        D * (Dxx(u(x, y, z, t)) + Dyy(u(x, y, z, t)) + Dzz(u(x, y, z, t))) + S(x, y, z, t)

    eqs = [eq_mass, eq_mom_x, eq_mom_y, eq_mom_z, eq_energy, eq_transport]

    # Dominio adimensional 3D del Valle
    domains = [
        x ∈ DomainSets.ClosedInterval(-1.0, 1.0),
        y ∈ DomainSets.ClosedInterval(-1.0, 1.0),
        z ∈ DomainSets.ClosedInterval(0.0, 1.0),
        t ∈ DomainSets.ClosedInterval(0.0, 1.0)
    ]

    # Condiciones de frontera (Actualizadas para 3D)
    bcs = [
        # Paredes del valle (Este-Oeste x = ±1) - Vientos cero
        vx(-1.0, y, z, t) ~ 0.0, vx(1.0, y, z, t) ~ 0.0,
        vy(-1.0, y, z, t) ~ 0.0, vy(1.0, y, z, t) ~ 0.0,
        vz(-1.0, y, z, t) ~ 0.0, vz(1.0, y, z, t) ~ 0.0,
        
        # Extremos del valle (Sur-Norte y = ±1) - Permeables (no imponemos 0, dejamos libre o fijo si se conoce, por ahora 0 simplificado)
        vx(x, -1.0, z, t) ~ 0.0, vx(x, 1.0, z, t) ~ 0.0,
        vy(x, -1.0, z, t) ~ 0.0, vy(x, 1.0, z, t) ~ 0.0,
        vz(x, -1.0, z, t) ~ 0.0, vz(x, 1.0, z, t) ~ 0.0,

        # Fondo del valle (z = 0)
        vx(x, y, 0.0, t) ~ 0.0,
        vy(x, y, 0.0, t) ~ 0.0,
        vz(x, y, 0.0, t) ~ 0.0,

        # Temperatura (Fondo frío, Techo cálido)
        T(x, y, 0.0, t) ~ -1.0, 
        T(x, y, 1.0, t) ~ 1.0,  

        # Tapa de Inversión Térmica para el Material Particulado (No escapa PM2.5 por el techo)
        Dz(u(x, y, 1.0, t)) ~ 0.0,

        # Cero concentración en los bordes del dominio (fuera del valle)
        u(-1.0, y, z, t) ~ 0.0, u(1.0, y, z, t) ~ 0.0,
        u(x, -1.0, z, t) ~ 0.0, u(x, 1.0, z, t) ~ 0.0,

        # Condiciones Iniciales (t = 0)
        vx(x, y, z, 0.0) ~ 0.0,
        vy(x, y, z, 0.0) ~ 0.0,
        vz(x, y, z, 0.0) ~ 0.0,
        P(x, y, z, 0.0) ~ 0.0,
        T(x, y, z, 0.0) ~ z,    
        u(x, y, z, 0.0) ~ 0.0
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [x, y, z, t], [u(x, y, z, t), T(x, y, z, t), vx(x, y, z, t), vy(x, y, z, t), vz(x, y, z, t), P(x, y, z, t), S(x, y, z, t)])
    return pdesys, (x, y, z, t, u, T, vx, vy, vz, P, S)
end

"""
    build_multi_pinn()

Construye 7 redes neuronales para aproximar las variables 3D:
[u, T, vx, vy, vz, P, S]
"""
function build_multi_pinn()
    # 4 Entradas: x, y, z, t. 1 Salida.
    make_net = () -> Lux.Chain(
        Lux.Dense(4, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1)
    )

    # Red de emisión S con activación final softplus (S >= 0.0)
    net_s = Lux.Chain(
        Lux.Dense(4, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1, Lux.softplus)
    )

    return [make_net(), make_net(), make_net(), make_net(), make_net(), make_net(), net_s]
end

# ==============================================================================
# ESTRATEGIA DE MUESTREO DE IMPORTANCIA ESPACIAL (Propuesta 3 Adaptada a 3D)
# ==============================================================================
const QMC = NeuralPDE.QuasiMonteCarlo

struct ImportanceSampler <: QMC.SamplingAlgorithm end

function QMC.sample(n::Integer, d::Integer, sampler::ImportanceSampler, T::Type)
    base_samples = QMC.sample(n, d, QMC.LatinHypercubeSample(), T)
    if d == 4 # Espacio-tiempo 3D: [x, y, z, t]
        for i in 1:n
            # 1. Recuperar coordenadas físicas
            x_phys = 2.0 * base_samples[1, i] - 1.0
            
            # Simulamos el relieve (parábola en x). Más adelante se reemplaza por el DEM real h(x,y).
            h_xy = 0.4 * x_phys^2
            
            # 2. Aplicar muestreo de importancia bimodal vertical en z
            z_uni = base_samples[3, i]
            if rand() < 0.5
                z_new = z_uni^3.0 # Sesgo hacia el suelo
            else
                z_new = 0.5 + 4.0 * (z_uni - 0.5)^3.0 # Sesgo hacia la inversión térmica
            end
            
            # 3. Enmascaramiento de colocación
            cutoff = h_xy - 0.05
            if z_new < cutoff
                z_new = cutoff + z_uni * (1.0 - cutoff)
            end
            
            base_samples[3, i] = clamp(z_new, 0.0, 1.0)
        end
    end
    return base_samples
end

function QMC.sample(n::Integer, d::Integer, sampler::ImportanceSampler)
    return QMC.sample(n, d, sampler, Float64)
end

end # module
