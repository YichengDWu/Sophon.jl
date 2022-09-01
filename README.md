# Sophon

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MilkshakeForReal.github.io/Sophon.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MilkshakeForReal.github.io/Sophon.jl/dev/)
[![Build Status](https://github.com/MilkshakeForReal/Sophon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MilkshakeForReal/Sophon.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MilkshakeForReal/Sophon.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MilkshakeForReal/Sophon.jl)

`Sophon.jl` provides specialized neural networks and neural operators for Physics-informed machine learning.

## Installation

```julia
julia>] add "https://github.com/MilkshakeForReal/Sophon.jl"
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

f(x) = -4 * π^2 * sin(2*π*x)-250*π^2*sin(50*π*x)
eq = Dₓ²(u(x)) ~ f(x)
domain = [x ∈ 0..1]
bcs = [u(0) ~ 0, u(1) ~ 0]

@named possion = PDESystem(eq, bcs, domain, [x], [u(x)])

chain = Siren(1, (32,32,32,32,1))
discretization = PhysicsInformedNN(chain, GridTraining(1f-2))
prob = discretize(possion, discretization)

res = Optimization.solve(prob, Adam(5f-3), maxiters = 2000)

prob = remake(prob,u0=res.u)
res = Optimization.solve(prob, BFGS(), maxiters = 1000)

phi = discretization.phi
xs = 0:0.001:1
u_true = @. sin(2*pi * xs)+0.1*sin(50*pi*xs)
us = phi(xs',res.u)
fig = Figure()
axis = Axis(fig[1, 1])
lines!(xs, u_true, label = "Ground Truth")
lines!(xs, vec(us), label = "Prediction")
axislegend(axis)
display(fig)
```
![possion](https://github.com/MilkshakeForReal/Sophon.jl/blob/main/assets/poisson.png)

## Related Libraries

- [PaddleScience](https://github.com/PaddlePaddle/PaddleScience)
- [MindScience](https://gitee.com/mindspore/mindscience)
- [Modulus](https://docs.nvidia.com/deeplearning/modulus/index.html#)
- [DeepXDE](https://deepxde.readthedocs.io/en/latest/index.html#)
