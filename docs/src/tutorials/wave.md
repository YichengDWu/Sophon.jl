# 1D Wave Equation

This example is taken from [here](https://docs.nvidia.com/deeplearning/modulus/user_guide/foundational/1d_wave_equation.html).

The wave is described by the below equation.
```math
\begin{aligned}
u_{t t} &=c^{2} u_{x x} \\
u(0, t) &=0 \\
u(\pi, t) &=0 \\
u(x, 0) &=\sin (x) \\
u_{t}(x, 0) &=\sin (x)
\end{aligned}
```
where, the wave speed ``c=1`` and the analytical solution to the above problem is given by ``\sin (x)(\sin (t)+\cos (t))``.

```julia
using NeuralPDE, Lux, Random, Optimization, OptimizationOptimJL, OptimizationOptimisers, IntervalSets
using Random

@parameters t, x, c
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

eq = Dtt(u(x,t)) ~ Dxx(u(x,t))

# make domain
domain = [x ∈ Interval(0.0, π),
          t ∈ Interval(0.0, 2π)]

# boundary conditions
bcs = [u(x, 0) ~ sin(x),
       Dt(u(x, 0)) ~ sin(x),
       u(0, t) ~ 0,
       u(π, t) ~ 0]
@named pde_system = PDESystem(eq,bcs,domain,[x,t],[u(x,t)])


net = FullyConnected(2, 1, sin; num_layers = 5, hidden_dims = 32)
ps = Lux.setup(Random.default_rng(), net)[1] |> Lux.ComponentArray |> gpu .|> Float64
discretization = PhysicsInformedNN(net, QuasiRandomTraining(200); init_params = ps)
phi = discretization.phi

prob = discretize(pde_system, discretization)

opt = Scheduler(Adam(), Sophon.Step(λ = 1e-3, γ = 0.95, step_sizes=100))
@time res = Optimization.solve(prob,opt; maxiters=2000)
```
Let's visualize the results.

```julia
xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domain]
u_analytic(x,t) = sin(x)*(sin(t)+cos(t))
predict(x,t) = first(phi(gpu([x,t]),res.u))
u_real = u_analytic.(xs,ts')
u_pred = predict.(xs,ts')

using CairoMakie
axis = (xlabel="x", ylabel="t", title="Analytical Solution")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, u_real, axis=axis)
Colorbar(fig[:, end+1], hm1)
ax2, hm2= CairoMakie.heatmap(fig[1, end+1], xs, ts, u_pred, axis= merge(axis, (;title = "Prediction")))
Colorbar(fig[:, end+1], hm2)
ax3, hm3 = CairoMakie.heatmap(fig[1, end+1], xs, ts, u_pred-u_real, axis= merge(axis, (;title = "Error")))
Colorbar(fig[:, end+1], hm3)

fig
```

Let's see how causal training can help imporve the results.

```julia
discretization = PhysicsInformedNN(net, CausalTraining(200;epsilon = 20); init_params = ps)
phi = discretization.phi

prob = discretize(pde_system, discretization)
@time res = Optimization.solve(prob,opt; maxiters=2000)
```

```julia; echo = false
xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domain]
u_analytic(x,t) = sin(x)*(sin(t)+cos(t))
predict(x,t) = first(phi([x,t],res.u))
u_real = u_analytic.(xs,ts')
u_pred = predict.(xs,ts')
using CairoMakie
axis = (xlabel="x", ylabel="t", title="Analytical Solution")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, u_real, axis=axis)
Colorbar(fig[:, end+1], hm1)
ax2, hm2= CairoMakie.heatmap(fig[1, end+1], xs, ts, u_pred, axis= merge(axis, (;title = "Prediction")))
Colorbar(fig[:, end+1], hm2)
ax3, hm3 = CairoMakie.heatmap(fig[1, end+1], xs, ts, u_pred-u_real, axis= merge(axis, (;title = "Error")))
Colorbar(fig[:, end+1], hm3)

fig
```