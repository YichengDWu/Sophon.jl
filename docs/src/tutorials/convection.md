# 1D Convection Equation

Consider the following 1D-convection equation with periodic boundary conditions.

```math
\begin{aligned}
&\frac{\partial u}{\partial t}+c \frac{\partial u}{\partial x}=0, x \in[0,1], t \in[0,1] \\
&u(x, 0)=sin(2\pi x) \\
\end{aligned}
```

First we define the PDE.

```@example convection
using NeuralPDE, Lux, Random, Sophon, IntervalSets, CairoMakie
using Optimization, OptimizationOptimJL, OptimizationOptimisers

@parameters x, t
@variables u(..)
Dₜ = Differential(t)
Dₓ = Differential(x)

c = 6
eq = Dₜ(u(x,t)) + c * Dₓ(u(x,t)) ~ 0
u_analytic(x,t) = sinpi(2*(x-c*t))

domains = [x ∈ 0..1, t ∈ 0..1]

bcs = [u(x,0) ~ u_analytic(x,0)]

@named convection = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])
```
## Imposing periodic boundary conditions
We will use [`BACON`](@ref) to impose the boundary conditions. To this end, we simply set `period` to be one.

```@example convection
chain = BACON(2,1; hidden_dims = 32, num_layers=5, period = 1, N = 6)
```

!!! note
    For demonstration purposes, the model is also periodic in time

```@example convection
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(500; resampling = false, minibatch = 1); adaptive_loss = NonAdaptiveLoss(; bc_loss_weights = [100]))

prob = discretize(convection, discretization) 

@time res = Optimization.solve(prob, LBFGS(); maxiters = 500)
```

Let's visualize the result.

```@example convection
phi = discretization.phi

fig, ax, hm = CairoMakie.heatmap(ts, xs, u_pred', axis=(xlabel="t", ylabel="x", title="c = $c"))
ax2, hm2 = heatmap(fig[1,end+1], ts,xs, abs.(u_pred' .- u_real'), axis = (xlabel="t", ylabel="x", title="Absolute error"))
Colorbar(fig[:, end+1], hm2)
save("convection.png", fig); nothing # hide
```
![](convection.png)

We can verify that our model is indeed, periodic.

```@example convection
xs, ts= [infimum(d.domain):0.01:supremum(d.domain)*2 for d in domains]
u_pred = [sum(phi([x,t],res.u)) for x in xs, t in ts]
fig, ax, hm = CairoMakie.heatmap(ts, xs, u_pred', axis=(xlabel="t", ylabel="x", title="c = $c"))
save("convection2.png", fig); nothing # hide
```
![](convection2.png)

## Respecting causality

[`CausalTraining`](@ref) will only start optimizing the loss of the succeeding time after the loss of the preceding time has been optimized.

```@example convectio
strategy =  CausalTraining(500; init_points = 200, epsilon = 1, bc_loss_weights = [100])

global i = 0
function callback(p,l)
    @show l
    global i
    i = i+1
    i > 200 && (strategy.reweight = true)
    phi = discretization.phi

    u_pred = [sum(phi([x,t],p)) for x in xs, t in ts]
    
    fig, ax, hm = CairoMakie.heatmap(ts, xs, u_pred', axis=(xlabel="t", ylabel="x", title="c = #$c"))
    #fig = plot(vec(strategy.W))
    display(fig)
    return false
end
discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(convection, discretization) 

@time res = Optimization.solve(prob, LBFGS(); maxiters = 500, callback = callback)

phi = discretization.phi

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_pred = [sum(phi([x,t],res.u)) for x in xs, t in ts]
u_real = u_analytic.(xs,ts')

fig, ax, hm = CairoMakie.heatmap(ts, xs, u_pred', axis=(xlabel="t", ylabel="x", title="c = $c"))
ax2, hm2 = heatmap(fig[1,end+1], ts,xs, abs.(u_pred' .- u_real'), axis = (xlabel="t", ylabel="x", title="Absolute error"))
Colorbar(fig[:, end+1], hm2)

save("convection3.png", fig); nothing # hide
```
![](convection3.png)

!!! note
    The hyperparameter `epsilon` in [`CausalTraining`](@ref) is very sensitive.
