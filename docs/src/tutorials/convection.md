# 1D Convection Equation

Consider the following 1D-convection equation

```math
\begin{aligned}
&\frac{\partial u}{\partial t}+c \frac{\partial u}{\partial x}=0, x \in[0,1], t \in[0,1] \\
&u(x, 0)=sin(2\pi x) \\
&u(0,t) = -sin(2\pi ct)\\
&u(1,t) = -sin(2\pi ct)
\end{aligned}
```

where ``c = 50/2\pi``. First we solve it with `QuasiRandomTraining`.

```@example convection
using NeuralPDE, Lux, Random, Sophon, IntervalSets, CairoMakie
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using CUDA
CUDA.allowscalar(false)

@parameters x, t
@variables u(..)
Dₜ = Differential(t)
Dₓ = Differential(x)

β = 50
c = β/2π
eq = Dₜ(u(x,t)) + c * Dₓ(u(x,t)) ~ 0
u_analytic(x,t) = sin(2π*(x-c*t))

domains = [x ∈ 0..1, t ∈ 0..1]

bcs = [u(0,t) ~ u_analytic(0,t),
       u(1,t) ~ u_analytic(1,t),
       u(x,0) ~ u_analytic(x,0)]

@named convection = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])

chain = Siren(2, 1; num_layers = 5, hidden_dims = 50, omega = 5f0)
ps = Lux.initialparameters(Random.default_rng(), chain) |> GPUComponentArray64
discretization = PhysicsInformedNN(chain, QuasiRandomTraining(100); init_params=ps, adaptive_loss = NonAdaptiveLoss(pde_loss_weights = 1, bc_loss_weights = 100), order = 2)
prob = discretize(convection, discretization)

@time res = Optimization.solve(prob, Adam(); maxiters = 3000)
```

Let's visualize the result.

```@example convection
phi = discretization.phi

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_pred = [sum(phi(gpu([x,t]),res.u)) for x in xs, t in ts]
u_real = u_analytic.(xs,ts')

axis = (xlabel="t", ylabel="x", title="β = $β")
fig, ax, hm = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)
ax2, hm2 = heatmap(fig[1,end+1], ts,xs, abs.(u_pred' .- u_real'), axis = (xlabel="t", ylabel="x", title="error"))
Colorbar(fig[:, end+1], hm2)

save("convection.png", fig); nothing # hide
```
![](convection.png)

## Compared to Method of Lines

```@example convection
using MethodOfLines
dx = 0.001
order = 4
mol_discretization = MOLFiniteDifference([x => dx], t, approx_order = order)

# Convert the PDE problem into an ODE problem
prob = discretize(convection,mol_discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat=0.001)

grid = get_discrete(convection, mol_discretization)
discrete_x = grid[x]
discrete_t = sol[t]

solu = [map(d -> sol[d][i], grid[u(x, t)]) for i in 1:length(sol[t])]
u_pred = hcat(solu...)

fig_, ax, hm = CairoMakie.heatmap(ts, xs, u_pred', axis=axis)
ax2, hm2 = heatmap(fig_[1,end+1], ts,xs, abs.(u_pred' .- u_analytic.(discrete_x, discrete_t')'), axis = (xlabel="t", ylabel="x", title="error"))
Colorbar(fig_[:, end+1], hm2)
save("convection2.png", fig_); nothing # hide
```
![](convection2.png)
