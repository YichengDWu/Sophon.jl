# Sophon

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yichengdwu.github.io/Sophon.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yichengdwu.github.io/Sophon.jl/dev/)
[![Build Status](https://github.com/YichengDWu/Sophon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YichengDWu/Sophon.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YichengDWu/Sophon.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YichengDWu/Sophon.jl)
[![DOI](https://zenodo.org/badge/521846679.svg)](https://zenodo.org/badge/latestdoi/521846679)

`Sophon.jl` provides specialized neural networks and neural operators for Physics-informed machine learning.

Please star this repo if you find it useful.

## Installation

```julia
julia>] add Sophon
```

## Example: 1D Multi-scale Poisson Equation 
Simply replace primitive fully connected neural nets with those defined in this pacakge!

```julia
using NeuralPDE, IntervalSets, Lux, Sophon
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using CairoMakie

@parameters x
@variables u(..)
Dₓ² = Differential(x)^2

f(x) = -4 * π^2 * sin(2 * π * x) - 250 * π^2 * sin(50 * π * x)
eq = Dₓ²(u(x)) ~ f(x)
domain = [x ∈ 0 .. 1]
bcs = [u(0) ~ 0, u(1) ~ 0]

@named poisson = PDESystem(eq, bcs, domain, [x], [u(x)])

chain = Siren(1, 32, 32, 32, 32, 1)
adaptive_loss = NonAdaptiveLoss(;bc_loss_weights=[100,100])
discretization = PhysicsInformedNN(chain,  GridTraining(0.01); adaptive_loss = adaptive_loss)
prob = discretize(poisson, discretization)

res = Optimization.solve(prob, Adam(5.0f-3); maxiters=2000)

prob = remake(prob; u0=res.u)
res = Optimization.solve(prob, LBFGS(); maxiters=1000)

using CairoMakie
phi = discretization.phi
xs = 0:0.001:1
u_true = @. sin(2 * pi * xs) + 0.1 * sin(50 * pi * xs)
us = phi(xs', res.u)
fig = Figure()
axis = Axis(fig[1, 1])
lines!(xs, u_true; label="Ground Truth")
lines!(xs, vec(us); label="Prediction")
axislegend(axis)
display(fig)
```
![possion](https://github.com/YichengDWu/Sophon.jl/blob/main/assets/poisson.png)

## Related Libraries

- [PaddleScience](https://github.com/PaddlePaddle/PaddleScience)
- [MindScience](https://gitee.com/mindspore/mindscience)
- [Modulus](https://docs.nvidia.com/deeplearning/modulus/index.html#)
- [DeepXDE](https://deepxde.readthedocs.io/en/latest/index.html#)
