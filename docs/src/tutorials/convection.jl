using NeuralPDE, Lux, Random, Sophon, IntervalSets
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using CUDA
CUDA.allowscalar(false)

@parameters x, t
@variables u(..)
Dₜ = Differential(t)
Dₓ = Differential(x)

β = 50
c = β/2π
eq = Dₜ(u(x,t)) + c * Dₓ(u(x,t)) ~ 0
u_analytic(x,t) = sin(2π*(x-c*t))

domains = [x ∈ 0..1, t ∈ 0..1]

bcs = [u(0,t) ~ u_analytic(0,t),
       u(1,t) ~ u_analytic(1,t),
       u(x,0) ~ sin(2π*x)]

@named convection = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])

chain = FullyConnected(2, 1, tanh; num_layers = 5, hidden_dims = 50)
ps = Lux.initialparameters(Random.default_rng(), chain) |> GPUComponentArray64
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(100); init_params=ps)
prob = discretize(convection, discretization)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

@time res = Optimization.solve(prob, Adam(); maxiters = 3000, callback = callback)
phi = discretization.phi

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
predict(x,t) = sum(phi(gpu([x,t]),res.u))
u_pred = predict.(xs,ts')

using CairoMakie
axis = (xlabel="t", ylabel="x", title="β = $β without causal training")
fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)

###############################################################################

discretization = PhysicsInformedNN(chain, CausalTraining(100; epsilon = 150); init_params=ps)
phi = discretization.phi

prob = discretize(convection, discretization)

@time res = Optimization.solve(prob, Adam(); maxiters = 3000, callback = callback)

predict(x,t) = sum(phi(gpu([x,t]),res.u))
u_pred = predict.(xs,ts')

axis = (xlabel="t", ylabel="x", title="β = $β with causal training")
fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)
