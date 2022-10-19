using Sophon, ModelingToolkit
using DomainSets
using DomainSets: ×
using Optimization, OptimizationOptimJL

@parameters x t
@variables u(..) a(..)
Dₓ = Differential(x)
Dₜ = Differential(t)
Dₓ² = Dₓ^2

ν = 0.001
eq = Dₜ(u(x,t)) + u(x,t) * Dₓ(u(x,t)) ~ ν * Dₓ²(u(x,t))
domain = (0.0 .. 1.0) × (0.0 .. 1.0)
eq = eq => domain

bcs = [(u(0.0, t) ~ u(1.0, t)) => (0.0 .. 1.0),
       (u(x, 0.0) ~ a(x)) => (0.0 .. 1.0) × (0.0 .. 0.0)]

boundary = 0.0 .. 1.0

Burgers = Sophon.ParametricPDESystem([eq], bcs, [t, x], [u(x,t)], [a(x)])

chain = DeepONet((50, 50, 50, 50), tanh, (2, 50, 50, 50, 50), tanh)
pinn = PINN(chain)
sampler = QuasiRandomSampler(500, 50)
strategy = NonAdaptiveTraining(1, (1, 20))

struct MyFuncSampler <: Sophon.FunctionSampler
    num_fcs::Int
end

Sophon.sample(::MyFuncSampler) = [cospi, sinpi, x -> cospi(2x), x-> sinpi(2x)]

cord_branch_net = [range(0.0, 1.0, length=50)...]

prob = Sophon.discretize(Burgers, pinn, sampler, strategy, MyFuncSampler(5), cord_branch_net)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

@time res = Optimization.solve(prob, BFGS(); maxiters=500, callback=callback)

using CairoMakie

phi = pinn.phi
xs = 0.0:0.001:1.0
ts = 0.0:0.001:1.0

u0 = reshape(cospi.(cord_branch_net), :, 1)
axis = (xlabel="t", ylabel="x", title="Prediction")
u_pred = [sum(pinn.phi((u0, [x, t]), res.u)) for x in xs, t in ts]
fig, ax, hm = heatmap(ts, xs, u_pred', axis=axis, colormap=:jet)
