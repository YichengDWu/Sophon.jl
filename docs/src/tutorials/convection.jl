using NeuralPDE, Lux, Random, Sophon, IntervalSets
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using CUDA
CUDA.allowscalar(false)

@parameters x, t
@variables u(..)
Dₜ = Differential(t)
Dₓ = Differential(x)

β = 10
eq = Dₜ(u(x,t)) + β * Dₓ(u(x,t)) ~ 0

domains = [x ∈ 0..1, t ∈ 0..1]
bcs = [u(0,t) ~ sin(-2π*β*t),
       u(1,t) ~ sin(1-2π*β*t),
       u(x,0) ~ sin(2π*x)]

@named convection = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])

chain = FullyConnected(2, 1, tanh; num_layers = 5, hidden_dims = 50)
ps = Lux.initialparameters(Random.default_rng(), chain) |> GPUComponentArray64
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(500); init_params=ps)
prob = discretize(convection, discretization)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

@time res = Optimization.solve(prob, LBFGS(); maxiters = 1000, callback = callback)
phi = discretization.phi

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
predict(x,t) = sum(phi(gpu([x,t]),res.u))
u_pred = predict.(xs,ts')

using CairoMakie
axis = (xlabel="t", ylabel="x", title="β = $β")
fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)

###############################################################################

discretization = PhysicsInformedNN(chain, CausalTraining(500;epsilon = 0); init_params=ps)
prob = discretize(convection, discretization)

@time res = Optimization.solve(prob, LBFGS(); maxiters = 1000, callback = callback)

predict(x,t) = sum(phi(gpu([x,t]),res.u))
u_pred = predict.(xs,ts')

fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)
