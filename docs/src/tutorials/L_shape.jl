using ModelingToolkit, DomainSets
using DomainSets: ×
using Sophon

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -1.0
eqs = [(eq, (-1..0) × (-1..1)),
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
