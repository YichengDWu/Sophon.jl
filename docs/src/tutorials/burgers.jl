using Sophon, ModelingToolkit
using DomainSets
using DomainSets: ×
using Optimization, OptimizationOptimJL
import OptimizationFlux: Adam
using Interpolations, GaussianRandomFields
using Setfield

@parameters x t
@variables u(..) a(..)
Dₓ = Differential(x)
Dₜ = Differential(t)
Dₓ² = Dₓ^2

ν = 0.001
eq = Dₜ(u(x, t)) + u(x, t) * Dₓ(u(x, t)) ~ ν * Dₓ²(u(x, t))
domain = (0.0 .. 1.0) × (0.0 .. 1.0)
eq = eq => domain

bcs = [
    (u(0.0, t) ~ u(1.0, t)) => (0.0 .. 1.0),
    (u(x, t) ~ a(x)) => (0.0 .. 1.0) × (0.0 .. 0.0),
]

boundary = 0.0 .. 1.0

Burgers = Sophon.ParametricPDESystem([eq], bcs, [t, x], [u(x, t)], [a(x)])

chain = DeepONet((50, 50, 50, 50), tanh, (2, 50, 50, 50, 50), tanh)
pinn = PINN(chain)
sampler = QuasiRandomSampler(500, 50)
strategy = NonAdaptiveTraining()

struct MyFuncSampler <: Sophon.FunctionSampler
    pts::Any
    grf::Any
    n::Any
end

function MyFuncSampler(pts, n)
    cov = CovarianceFunction(1, Whittle(0.1))
    grf = GaussianRandomField(cov, KarhunenLoeve(5), pts)
    return MyFuncSampler(pts, grf, n)
end

function Sophon.sample(sampler::MyFuncSampler)
    (; n, grf, pts) = sampler
    ys = [cubic_spline_interpolation(pts, pts .* (1 .- pts) .* sample(grf) .+ 0.1*randn())]
    for _ in 1:n-1
        y = cubic_spline_interpolation(pts, pts .* (1 .- pts) .* sample(grf) .+ 0.1*randn())
        push!(ys, y)
    end
    return ys
end

cord_branch_net = range(0.0, 1.0; length=50)

fsampler = MyFuncSampler(cord_branch_net, 10)

prob = Sophon.discretize(Burgers, pinn, sampler, strategy, fsampler, cord_branch_net)

function callback(p, l)
    println("Loss: $l")
    return false
end

@time res = Optimization.solve(prob, BFGS(); maxiters=1000, callback=callback)

using ProgressMeter

n = 20000
k = 10
pg = Progress(n; showspeed=true)

function callback(p, l)
    ProgressMeter.next!(pg; showvalues=[(:loss, l)])
    return false
end

adam = Adam()
for i in 1:k
    cords = Sophon.sample(Burgers, sampler, strategy)
    fs = Sophon.sample(fsampler)
    p = Sophon.PINOParameterHandler(cords, fs)
    prob = remake(prob; u0=res.u, p=p)
    res = Optimization.solve(prob, adam; maxiters=n ÷ k, callback=callback)
end

using CairoMakie

phi = pinn.phi
xs = 0.0:0.01:1.0
ts = 0.0:0.01:1.0

f_test(x) = sinpi(2x)
u0 = reshape(f_test.(cord_branch_net), :, 1)
axis = (xlabel="t", ylabel="x", title="Prediction")
u_pred = [sum(pinn.phi((u0, [x, t]), res.u)) for x in xs, t in ts]
fig, ax, hm = heatmap(ts, xs, u_pred'; axis=axis, colormap=:jet)
