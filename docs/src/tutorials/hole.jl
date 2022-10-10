
using ModelingToolkit, Sophon
using Optimization, OptimizationOptimJL
using DomainSets
using DomainSets: ×

@parameters x, y
@variables T(..)

Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

r = 0.2
l = 1.0

eq = (Dxx(T(x,y)) + Dyy(T(x,y)) -4)*sqrt(abs2(x)+abs2(y)) + 2r ~ 0.0
eq = (eq, (-l .. l) × (-l .. l) \ Disk(r))

bcs = [(T(x,y) ~ 0.0, r*UnitCircle()),
       (T(x,y) ~ abs2(sqrt(abs2(l) + abs2(y)) - r), (-l .. -l) × (-l .. l)),
       (T(x,y) ~ abs2(sqrt(abs2(l) + abs2(y)) - r), (l .. l) × (-l .. l)),
       (Dy(T(x,y)) ~ 2l*(2r / sqrt(abs2(l) + abs2(x)) - 1), (-l .. l) × (-l .. -l)),
       (Dy(T(x,y)) ~ -2l*(2r / sqrt(abs2(l) + abs2(x)) - 1), (-l .. l) × (l .. l))]

pde = Sophon.PDESystem(eq, bcs, [x, y], [T(x,y)])

chain = FullyConnected((2, 16, 16, 16, 16, 1), tanh)
pinn = PINN(chain)

sampler = QuasiRandomSampler(500, 100)
strategy = NonAdaptiveTraining(1,10)

prob = Sophon.discretize(pde, pinn, sampler, strategy)

function callback(p, l)
    println("Loss: $l")
    return false
end

@time res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 5000)

using CairoMakie

xs = range(-l, l, length = 200)
ys = range(-l, l, length = 200)

T_analytical(x, y) = abs2(sqrt(abs2(x) + abs2(y)) - r)
T_true = [[x,y] ∈ Disk(r) ? NaN : T_analytical(x, y) for x in xs, y in ys]
T_pred = [[x,y] ∈ Disk(r) ? NaN : pinn.phi([x, y], res.u)[1] for x in xs, y in ys]

fig, ax, hm = heatmap(xs, ys, T_pred)
fig, ax, hm = heatmap(xs, ys, T_true)

fig, ax, hm = heatmap(xs, ys, abs.(T_pred .- T_true))
Colorbar(fig[:, end+1], hm)

fig
