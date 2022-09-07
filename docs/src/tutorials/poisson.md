# 1D Poisson's Equation

This example is taken from [wang2021eigenvector](@cite). Consider a simple 1D Poisson’s equation with Dirichlet boundary conditions. The solution is given by

```math
u(x)=\sin (2 \pi x)+0.1 \sin (50 \pi x)
```

```@example poisson
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
```

```@example poisson
chain = Siren(1, 32, 32, 32, 32, 1)
adaptive_loss = NonAdaptiveLoss(;bc_loss_weights=[100,100])
discretization = PhysicsInformedNN(chain,  GridTraining(0.01); adaptive_loss = adaptive_loss)
prob = discretize(poisson, discretization)

res = Optimization.solve(prob, Adam(5.0f-3); maxiters=2000)

prob = remake(prob; u0=res.u)
res = Optimization.solve(prob, LBFGS(); maxiters=1000)

phi = discretization.phi
xs = 0:0.001:1
u_true = @. sin(2 * pi * xs) + 0.1 * sin(50 * pi * xs)
us = phi(xs', res.u)
fig = Figure()
axis = Axis(fig[1, 1])
lines!(xs, u_true; label="Ground Truth")
lines!(xs, vec(us); label="Prediction")
axislegend(axis)
save("result.png", fig); nothing # hide
```

![](result.png)

## Compute the relative L2 error

```@example poisson
using Integrals

u_analytical(x,p) = sin.(2 * pi .* x) + 0.1 * sin.(50 * pi .* x)
error(x,p) = abs2.(vec(phi([x;;],res.u)) .- u_analytical(x,p))

relative_L2_error = solve(IntegralProblem(error,0,1),HCubatureJL(),reltol=1e-3,abstol=1e-3) ./ solve(IntegralProblem((x,p) -> abs2.(u_analytical(x,p)),0, 1),HCubatureJL(),reltol=1e-3,abstol=1e-3)
```
