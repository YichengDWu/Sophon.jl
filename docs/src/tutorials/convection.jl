using NeuralPDE, Lux, Random, Sophon, IntervalSets
using Optimization, OptimizationOptimJL, OptimizationOptimisers

@parameters x, t
@variables u(..)
Dₜ = Differential(t)
Dₓ = Differential(x)

β = 10
eq = Dₜ(u(x,t)) + β * Dₓ(u(x,t)) ~ 0

domains = [x ∈ 0..2π, t ∈ 0..1]
bcs = [u(0,t) ~ u(2π,t), u(x,0) ~ sin(x)]

@named convection = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])

#chain = Siren(2,(32,32,32,1))
chain = FullyConnected(2,(50,50,50,50,1), tanh)
discretization = PhysicsInformedNN(chain, GridTraining(0.1))
prob = discretize(convection, discretization)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, Adam(); maxiters = 2000, callback = callback)
phi = discretization.phi

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
predict(x,t) = first(phi([x,t],res.u))
u_pred = predict.(xs,ts')

using CairoMakie
axis = (xlabel="t", ylabel="x", title="Analytical Solution")
fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)
