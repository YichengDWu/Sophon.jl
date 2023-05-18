# Getting started with ODEs

This tutorial provides a step-by-step guide to solve the Lotka-Volterra system of ordinary differential equations (ODEs).

```@example ODE
using ModelingToolkit
using Sophon, IntervalSets
using Optimization, OptimizationOptimJL
using Plots

# Defining parameters and variables
@parameters t
@variables x(..), y(..)

# Define the differential operator
Dₜ = Differential(t)
p = [1.5, 1.0, 3.0, 1.0]

# Setting up the system of equations
eqs = [Dₜ(x(t)) ~ p[1] * x(t) - p[2] * x(t) * y(t),
      Dₜ(y(t)) ~ -p[3] * y(t) + p[4] * x(t) * y(t)]

# Defining the domain
domain = [t ∈ 0 .. 3.0]

# Defining the initial conditions
bcs = [x(0.0) ~ 1.0, y(0.0) ~ 1.0]

@named lotka_volterra = PDESystem(eqs, bcs, domain, [t], [x(t), y(t)])
```

In this part of the tutorial, we will employ [`BetaRandomSampler`](@ref) to generate training data using a Beta distribution. This introduces a soft causality into the training data, enhancing the effectiveness of the learning process.

```@example ODE
# Constructing the physics-informed neural network (PINN)
pinn = PINN(x = FullyConnected(1, 1, sin; hidden_dims=8, num_layers=3),
            y = FullyConnected(1, 1, sin; hidden_dims=8, num_layers=3))

# Setting up the sampler, training strategy and problem
sampler = BetaRandomSampler(200, 1)
strategy = NonAdaptiveTraining(1,50)
prob = Sophon.discretize(lotka_volterra, pinn, sampler, strategy)

# Solving the problem using BFGS optimization
res = Optimization.solve(prob, BFGS(); maxiters=1000)
```
Next, we'll compare our results with a reference solution to verify our computations.

```@example ODE
using OrdinaryDiffEq

function f(u, p, t)
    return [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2]]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
true_sol = solve(prob_oop, Tsit5(), saveat=0.01)

phi = pinn.phi
ts = [true_sol.t...;;]
x_pred = phi.x(ts, res.u.x)
y_pred = phi.y(ts, res.u.y)

plot(vec(ts), vec(x_pred), label="x_pred")
plot!(vec(ts), vec(y_pred), label="y_pred")
plot!(true_sol)
```

While the initial results are encouraging, we can further refine our model. By remaking the sampler, we can gradually transition to the uniform distribution for improved results.

```@example ODE
# Adjusting the sampler to uniform distribution and re-solving the problem
for α in [0.6, 0.8, 1.0] # when α = 1.0, it is equivalent to uniform sampling
    sampler = remake(sampler; α=α)
    data = Sophon.sample(lotka_volterra, sampler)
    prob = remake(prob; p=data, u0=res.u)
    res = Optimization.solve(prob, BFGS(); maxiters=1000)
end

# Generating new predictions and calculating the absolute error
x_pred = phi.x(ts, res.u.x)
y_pred = phi.y(ts, res.u.y)
maximum(sum(abs2, vcat(x_pred, y_pred) .- stack(true_sol.u); dims=1)) # print the absolute error
```
