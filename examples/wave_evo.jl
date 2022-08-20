
using NeuralPDE, Lux, Random, Optimization, OptimizationOptimJL, OptimizationOptimisers, IntervalSets
using Random, Sophon

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


net = Siren(2, 1; num_layers = 5, hidden_dims = 32, omega = 1.0)
ps = Lux.initialparameters(Random.default_rng(), net) |> GPUComponentArray64
discretization = PhysicsInformedNN(net, EvoTraining(200); init_params = ps)
phi = discretization.phi

prob = discretize(pde_system, discretization)

@time res = Optimization.solve(prob,Adam(); maxiters= 500)


xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in domain]
u_analytic(x,t) = sin(x)*(sin(t)+cos(t))
u_real = u_analytic.(xs,ts')

u_pred = [first(phi(gpu([x,t]),res.u)) for x in xs, t in ts]
using CairoMakie
axis = (xlabel="x", ylabel="t", title="Analytical Solution")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, u_real, axis=axis)
Colorbar(fig[:, end+1], hm1)
ax2, hm2= CairoMakie.heatmap(fig[1, end+1], xs, ts, u_pred, axis= merge(axis, (;title = "Prediction")))
Colorbar(fig[:, end+1], hm2)
ax3, hm3 = CairoMakie.heatmap(fig[1, end+1], xs, ts, abs.(u_pred-u_real), axis= merge(axis, (;title = "Absolute Error")))
Colorbar(fig[:, end+1], hm3)

fig
