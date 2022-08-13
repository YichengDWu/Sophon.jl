# 1D Poisson's Equation

```@example
using NeuralPDE, IntervalSets, Lux, Sophon
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using CairoMakie

@parameters x
@variables u(..)
Dxx = Differential(x)^2

f(x) = -4 * π^2 * sin(2 * π * x) - 250 * π^2 * sin(50 * π * x)
eq = Dxx(u(x)) ~ f(x)
domain = [x ∈ 0 .. 1]
bcs = [u(0) ~ 0, u(1) ~ 0]

@named possion = PDESystem(eq, bcs, domain, [x], [u(x)])

chain = Siren(1, (32, 32, 32, 32, 1))
discretization = PhysicsInformedNN(chain, GridTraining(1.0f-2))
prob = discretize(possion, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end
res = Optimization.solve(prob, Adam(5.0f-3); maxiters=1000, callback=callback)

prob = remake(prob; u0=res.u)
res = Optimization.solve(prob, LBFGS(); maxiters=500, callback=callback)

using CairoMakie
phi = discretization.phi
xs = 0:0.001:1
u_true = @. sin(2 * pi * xs) + 0.1 * sin(50 * pi * xs)
us = phi(xs', res.u)
fig = Figure()
axis = Axis(fig[1, 1])
lines!(xs, u_true; label="Ground truth")
lines!(xs, vec(us); label="prediction")
axislegend(axis)
save("result.png", fig); nothing # hide
```

![](result.png)

