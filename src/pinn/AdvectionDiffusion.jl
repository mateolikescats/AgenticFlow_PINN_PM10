module AdvectionDiffusion

using ModelingToolkit
using NeuralPDE
using Lux
using Optimization
using OptimizationOptimJL

export get_pde_system, build_pinn, parabolic_wind

"""
    parabolic_wind(x, V_max)

Modelo de viento acanalado para el Valle de Aburrá.
Asume que el eje Y fluye a lo largo del valle (Sur-Norte) y el eje X 
es transversal, donde x=-1 y x=1 son las paredes montañosas.
El viento es máximo en el centro (x=0) y nulo en las paredes.
"""
function parabolic_wind(x, V_max=1.0)
    return V_max * (1.0 - x^2)
end

"""
    get_pde_system(D_val, V_max)

Define simbólicamente la EDP de Advección-Difusión-Reacción.
Retorna el sistema PDE de ModelingToolkit.
"""
function get_pde_system(D_val=0.01, V_max=1.0)
    @parameters x y t
    @variables u(..)
    
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # Parámetros físicos (por ahora constantes o funciones dadas, 
    # en la Fase Inversa se volverán parámetros aprendibles)
    D = D_val
    vy = parabolic_wind(x, V_max)
    vx = 0.0 # Asumimos flujo laminar sin corrientes transversales
    
    # Ecuación: ∂u/∂t + v_x ∂u/∂x + v_y ∂u/∂y = D(∂²u/∂x² + ∂²u/∂y²) + S
    # Para la fase interpolativa, asumiremos S=0 (fuentes bases implícitas en los datos)
    S = 0.0
    
    eq = Dt(u(x,y,t)) + vx * Dx(u(x,y,t)) + vy * Dy(u(x,y,t)) ~ D * (Dxx(u(x,y,t)) + Dyy(u(x,y,t))) + S
    
    # Dominio adimensional
    domains = [
        x ∈ Interval(-1.0, 1.0),
        y ∈ Interval(-1.0, 1.0),
        t ∈ Interval(0.0, 1.0)
    ]
    
    # Condiciones de frontera iniciales (Dirichlet nulas en las montañas, 
    # en la práctica las condiciones vendrán guiadas por los datos de SIATA)
    bcs = [
        u(-1.0, y, t) ~ 0.0,
        u(1.0, y, t) ~ 0.0,
        u(x, -1.0, t) ~ 0.0, # Límite Sur
        u(x, 1.0, t) ~ 0.0,  # Límite Norte
        u(x, y, 0.0) ~ 0.0   # Condición Inicial (T=0)
    ]
    
    @named pdesys = PDESystem(eq, bcs, domains, [x,y,t], [u(x,y,t)])
    return pdesys, (x, y, t, u)
end

"""
    build_pinn()

Construye la arquitectura de la red neuronal que aproximará u(x,y,t).
"""
function build_pinn()
    # 3 Entradas: x, y, t. 1 Salida: Concentración u.
    # Usamos Tanh para garantizar que las segundas derivadas existan y sean suaves.
    chain = Lux.Chain(
        Lux.Dense(3, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1)
    )
    return chain
end

end # module
