module AdvectionDiffusion

using ModelingToolkit
using Lux
using DomainSets

export get_boussinesq_pde_system, build_multi_pinn

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
    
    # 1. Conservación de Masa (Incompresible)
    eq_mass = Dx(vx(x,z,t)) + Dz(vz(x,z,t)) ~ 0.0
    
    # 2. Momentum en X (Canalización transversal)
    eq_mom_x = Dt(vx(x,z,t)) + vx(x,z,t)*Dx(vx(x,z,t)) + vz(x,z,t)*Dz(vx(x,z,t)) ~ 
               -Dx(P(x,z,t)) + nu * (Dxx(vx(x,z,t)) + Dzz(vx(x,z,t)))
               
    # 3. Momentum en Z (Estratificación y Aproximación de Boussinesq)
    # El término beta_g * (T - T_ref) inyecta la fuerza de flotabilidad térmica.
    # En una inversión térmica, T cerca de z=0 es menor que en z>0, anulando la flotabilidad (vz <= 0).
    eq_mom_z = Dt(vz(x,z,t)) + vx(x,z,t)*Dx(vz(x,z,t)) + vz(x,z,t)*Dz(vz(x,z,t)) ~ 
               -Dz(P(x,z,t)) + nu * (Dxx(vz(x,z,t)) + Dzz(vz(x,z,t))) + beta_g * (T(x,z,t) - T_ref)
               
    # 4. Transporte de Calor (Radiación / Termodinámica)
    eq_energy = Dt(T(x,z,t)) + vx(x,z,t)*Dx(T(x,z,t)) + vz(x,z,t)*Dz(T(x,z,t)) ~ 
                alpha_T * (Dxx(T(x,z,t)) + Dzz(T(x,z,t)))
                
    # 5. Advección-Difusión de Contaminantes (PM2.5 / PM10)
    eq_transport = Dt(u(x,z,t)) + vx(x,z,t)*Dx(u(x,z,t)) + vz(x,z,t)*Dz(u(x,z,t)) ~ 
                   D * (Dxx(u(x,z,t)) + Dzz(u(x,z,t))) + S(x,z,t)
    
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
    # 3 Entradas: x, z, t. 1 Salida por cada variable.
    # Usamos redes separadas para evitar interferencia de gradientes entre variables físicas
    # de distintas escalas (ej. Presión vs Temperatura).
    make_net = () -> Lux.Chain(
        Lux.Dense(3, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1)
    )
    
    # Red de emisión S con activación final softplus para garantizar S >= 0.0 (físicamente consistente)
    net_s = Lux.Chain(
        Lux.Dense(3, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1, Lux.softplus)
    )
    
    return [make_net(), make_net(), make_net(), make_net(), make_net(), net_s]
end

end # module
