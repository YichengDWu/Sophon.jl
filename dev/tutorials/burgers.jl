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
       (u(x, t) ~ a(x)) => (0.0 .. 1.0) × (0.0 .. 0.0)]

boundary = 0.0 .. 1.0

Burgers = Sophon.ParametricPDESystem([eq], bcs, [t, x], [u(x,t)], [a(x)])

chain = DeepONet((50, 50, 50, 50), tanh, (2, 50, 50, 50, 50), tanh)
pinn = PINN(chain)
sampler = QuasiRandomSampler(500, 100)
strategy = NonAdaptiveTraining()

struct MyFuncSampler <: Sophon.FunctionSampler end

Sophon.sample(::MyFuncSampler) = [cospi, sinpi, x -> cospi(2x), x-> sinpi(2x), x -> 0.5*cospi(2x), x -> 0.5*sinpi(2x),
                                  x -> 0.25*cospi(x), x -> 0.25*sinpi(x), x -> 0.75*cospi(4x), x -> 0.75*sinpi(4x)]

cord_branch_net = range(0.0, 1.0, length=50) |> collect

prob = Sophon.discretize(Burgers, pinn, sampler, strategy, MyFuncSampler(), cord_branch_net)

function callback(p,l)
    println("Loss: $l")
    return false
end

@time res = Optimization.solve(prob, BFGS(); maxiters=1000, callback=callback)

using CairoMakie

phi = pinn.phi
xs = 0.0:0.01:1.0
ts = 0.0:0.01:1.0

f_test(x) = sinpi(x)
u0 = reshape(f_test.(cord_branch_net), :, 1)
axis = (xlabel="t", ylabel="x", title="Prediction")
u_pred = [sum(pinn.phi((u0, [x, t]), res.u)) for x in xs, t in ts]
fig, ax, hm = heatmap(ts, xs, u_pred', axis=axis, colormap=:jet)
