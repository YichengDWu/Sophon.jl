using Sophon, Test, ModelingToolkit, DomainSets
using Optimization, OptimizationOptimJL

@parameters x,t
@variables u(..), v(..)
Dₜ = Differential(t)
Dₓ² = Differential(x)^2

eqs=[Dₜ(u(x,t)) ~ -Dₓ²(v(x,t))/2 - (abs2(v(x,t)) + abs2(u(x,t))) * v(x,t),
     Dₜ(v(x,t)) ~  Dₓ²(u(x,t))/2 + (abs2(v(x,t)) + abs2(u(x,t))) * u(x,t)]

bcs = [u(x, 0.0) ~ 2sech(x),
       v(x, 0.0) ~ 0.0,
       u(-5.0, t) ~ u(5.0, t),
       v(-5.0, t) ~ v(5.0, t)]

domains = [x ∈ Interval(-5.0, 5.0),
           t ∈ Interval(0.0, π/2)]

@named pde_system = PDESystem(eqs, bcs, domains, [x,t], [u(x,t),v(x,t)])

pinn = PINN(u = Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0),
            v = Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0))

sampler = BetaRandomSampler(500, (200,200,20,20))
strategy = NonAdaptiveTraining(1,(10,10,1,1))

data = Sophon.sample(pde_system, sampler)

@test size(data[1]) == (2,500)
@test size(data[2]) == (2,500)
@test size(data[3]) == (2,200)
@test size(data[4]) == (2,200)
@test size(data[5]) == (2,20)
@test size(data[6]) == (2,20)

prob = Sophon.discretize(pde_system, pinn, sampler, strategy)
res = Optimization.solve(prob, BFGS(); maxiters=500)

@test res.objective < 1e-3

sampler = remake(sampler; α=0.8)
data = Sophon.sample(pde_system, sampler)
prob = remake(prob; p=data, u0=res.u)
res = Optimization.solve(prob, BFGS(); maxiters=500)

@test res.objective < 1e-3
