using NeuralPDE, Lux, Optimization, OptimizationOptimJL, Sophon, OptimizationOptimisers
import ModelingToolkit: Interval

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(t,x)) ~ C^2*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0
       u(t,1) ~ 0.,# for all t > 0
       u(0,x) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

# Neural network
chain = Lux.Chain(Dense(2,16,sin),Dense(16,16,sin),Dense(16,1))
discretization = PhysicsInformedNN(chain, CausalTraining(100;epsilon = 100))

@named pde_system = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
prob = discretize(pde_system,discretization)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

# optimizer
res = Optimization.solve(prob,Adam(); callback = callback, maxiters=1200)
phi = discretization.phi

opt = OptimizationOptimJL.BFGS()
