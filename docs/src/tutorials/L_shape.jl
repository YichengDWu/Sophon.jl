using ModelingToolkit, DomainSets, Optimization, OptimizationOptimJL
using DomainSets: ×
using Sophon

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -1.0
eqs = [(eq, (-1..0) × (-1..0)),
       (eq, (-1..0) × (0..1)),
       (eq, (0..1) × (-1..0))]

bc = u(x,y) ~ 0.0
boundaries = [(-1 .. -1) × (-1..1),
              (-1..0) × (1..1),
              (0..0) × (0..1),
              (0..1) × (0..0),
              (1..1) × (-1..0),
              (-1..1) × (-1 .. -1)]

bcs = [(bc, boundary) for boundary in boundaries]

pde_system = Sophon.PDESystem(eqs, bcs, [x,y], [u(x,y)])

chain = FullyConnected((2,16,16,16,1), tanh)
pinn = PINN(chain)
sampler = QuasiRandomSampler(500, 20)
strategy = NonAdaptiveTraining()

prob = Sophon.discretize(pde_system, pinn, sampler, strategy)

res = Optimization.solve(prob, BFGS(); maxiters=1000)

using CairoMakie

xs = -1:0.01:1
ys = -1:0.01:1

u_pred = [ifelse(x>0.0 && y>0.0, 0.0,pinn.phi([x,y], res.u)[1]) for x in xs, y in ys]
fig, ax, hm = heatmap(xs, ys, u_pred)
ColorBar(fig[1,2], hm)
