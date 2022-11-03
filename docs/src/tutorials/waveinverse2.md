# Inverse problem for the wave equation with unknown velocity field

We are going to sovle the wave equation.

```@example wave
using Sophon, ModelingToolkit, IntervalSets
using Optimization, OptimizationOptimJL

@parameters x, t
@variables u(..), c(..)

Dₜ = Differential(t)
Dₜ² = Differential(t)^2
Dₓ² = Differential(x)^2

s(x,t) = abs2(x) * sin(x) * cos(t)

eq = Dₜ²(u(x,t)) ~ c(x) * Dₓ²(u(x,t)) + s(x,t)

bcs = [u(x, 0) ~ sin(x),
       Dₜ(u(x, 0)) ~ 0,
       u(0, t) ~ 0,
       u(1, t) ~ sin(1) * cos(t)]

domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

@named wave = PDESystem(eq, bcs, domains, [t,x], [u(x,t),c(x)])
```

Here the velocity field ``c(x)`` is unknown, we will approximate it with a neural network.

```@example wave
pinn = PINN(u = FullyConnected((2,16,16,16,1), sin),
            c = FullyConnected((1,16,16,1), tanh))

sampler = QuasiRandomSampler(500,100)
strategy = NonAdaptiveTraining(1, (10,10,1,1))
```

Next we generate some data of ``u(x,t)``. Here we place two sensors at ``x=0.1`` and ``x=0.5``.

```@example wave
ū(x,t) = sin(x) * cos(t)

x_data = hcat(fill(0.1, 1, 50), fill(0.5, 1, 50))
t_data = repeat(range(0.0, 1.0, length = 50),2)'
input_data = [x_data; t_data]

u_data = ū.(x_data, t_data)
```
Finally we construct the inverse problem and solve it.

```@example wave
additional_loss(phi, θ) = sum(abs2, phi.u(input_data, θ.u) .- u_data)

prob = Sophon.discretize(wave, pinn, sampler, strategy; additional_loss=additional_loss)

@time res = Optimization.solve(prob, BFGS(), callback = callback, maxiters=1000)
```

Let's visualize the predictted solution and inferred velocity

```@example wave
using CairoMakie

ts = range(0, 1; length=100)
xs = range(0, 1; length=100)

u_pred = [pinn.phi.u([x, t], res.u.u)[1] for x in xs, t in ts]
c_pred = [pinn.phi.c([x], res.u.c)[1] for x in xs]

u_true = [ū(x, t) for x in xs, t in ts]
c_true = 1 .+ abs2.(xs) |> vec

axis = (xlabel="x", ylabel="t", title="Analytical Solution")
fig, ax1, hm1 = heatmap(xs, ts, u_true, axis=axis; colormap=:jet)
ax2, hm2= heatmap(fig[1, end+1], xs, ts, u_pred, axis= merge(axis, (;title = "Prediction")); colormap=:jet)
ax3, hm3 = heatmap(fig[1, end+1], xs, ts, abs.(u_true .- u_pred), axis= merge(axis, (;title = "Absolute Error")); colormap=:jet)
Colorbar(fig[:, end+1], hm3)
fig
save("sol.png", fig); nothing # hide
```
![](sol.png)


```@example wave
fig, ax = lines(xs, c_pred)
lines!(ax, xs, c_true)
fig
save("velocity.png", fig); nothing # hide
```
![](velocity.png)
