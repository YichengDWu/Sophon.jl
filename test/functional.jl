using ModelingToolkit, Sophon, DomainSets

@parameters x
@variables u(..), b(..)

f(x) = (1+(sin(π*x)^2))/(1+2*(sin(π*x)^2)) * π^2 * sin(π*x) + sin(π*x)
Dxx = Differential(x)^2
eq = [-b(u(x)) * Dxx(u(x)) + u(x) ~ f(x)]

bcs = [u(0) ~ 0, u(1) ~ 0]
domain = [x ∈ Interval(0.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domain, [x], [u(x), b(u(x))])

pinn = PINN(u = FullyConnected((1, 2, 1), tanh),
            b = FullyConnected((1, 2, 1), tanh))

sampler = QuasiRandomSampler(100)
strategy =  NonAdaptiveTraining()

prob = Sophon.discretize(pde_system, pinn, sampler, strategy)

@test prob.f(prob.u0, prob.p) isa AbstractFloat
