using ModelingToolkit, DomainSets, TaylorDiff, Sophon, Test
using Optimization, OptimizationOptimJL

@parameters x,t
@variables u(..), v(..)
Dₜ = Differential(t)
Dₓ² = Differential(x)^2

eqs=[Dₜ(u(x,t)) ~ -Dₓ²(v(x,t))/2 - (abs2(v(x,t)) + abs2(u(x,t))) * v(x,t),
     Dₜ(v(x,t)) ~  Dₓ²(u(x,t))/2 + (abs2(v(x,t)) + abs2(u(x,t))) * u(x,t)]

bcs = [u(x, 0.0f0) ~ 2sech(x),
       v(x, 0.0f0) ~ 0.0f0,
       u(-5.0f0, t) ~ u(5.0f0, t),
       v(-5.0f0, t) ~ v(5.0f0, t)]

domains = [x ∈ Interval(-5.0f0, 5.0f0),
           t ∈ Interval(0.0f0, π/2f0)]

@named pde_system = PDESystem(eqs, bcs, domains, [x,t], [u(x,t),v(x,t)])

derivative = isdefined(Base, :get_extension) ? Sophon.taylordiff : Sophon.SophonTaylorDiffExt.taylordiff

pinn = PINN(u = Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0),
            v = Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0))

sampler = QuasiRandomSampler(500, (200,200,20,20))
strategy = NonAdaptiveTraining(1,(10,10,1,1))

prob = Sophon.discretize(pde_system, pinn, sampler, strategy; derivative=derivative, fdtype=Float32)

@test eltype(prob.u0) == Float32
@test eltype(prob.p[1]) == Float32
@test prob.f(prob.u0, prob.p) isa Float32

res = Optimization.solve(prob, BFGS(); maxiters=1000)

@test res.objective < 1f-3
