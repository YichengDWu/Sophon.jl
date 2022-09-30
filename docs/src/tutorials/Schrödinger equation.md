# Schrödinger equation

The nonlinear Shrödinger equation is given by

$\mathrm{i} \partial_t \psi=-\frac{1}{2} \sigma \partial_{x x} \psi-\beta|\psi|^2 \psi$

Let $\sigma=\beta=1, \psi=u+v i$, the equation can be transformed to a system of partial differential equations

$u_t+\frac{1}{2} v_{x x}+\left(u^2+v^2\right) v=0$

$v_t-\frac{1}{2} u_{x x}-\left(u^2+v^2\right) u=0$

```@example Schrödinger
using ModelingToolkit, IntervalSets, Sophon, CairoMakie
using Optimization, OptimizationOptimJL, OptimizationOptimisers

@parameters x,t
@variables u(..), v(..)
Dₜ = Differential(t)
Dₓ² = Differential(x)^

eqs=[Dₜ(u(x,t)) ~ -Dₓ²(v(x,t))/2 - ((v(x,t))^2 + (u(x,t))^2) * v(x,t),
     Dₜ(v(x,t)) ~  Dₓ²(u(x,t))/2 + ((v(x,t))^2 + (u(x,t))^2) * u(x,t)]

bcs = [u(x, 0.0) ~ 2sech(x),
       v(x, 0.0) ~ 0.0,
       u(-5.0, t) ~ u(5.0, t),
       v(-5.0, t) ~ v(5.0, t)]

domains = [x ∈ Interval(-5.0, 5.0),
           t ∈ Interval(0.0, π/2)]

@named pde_system = PDESystem(eqs, bcs, domains, [x,t], [u(x,t),v(x,t)])
```

```@example Schrödinger
pinn = PINN(u=FullyConnected(2,1,tanh; hidden_dims=16,num_layers=3),
            v=FullyConnected(2,1,tanh; hidden_dims=16,num_layers=3))
            
sampler = QuasiRandomSampler(2000, (500,500,20,20))
strategy = NonAdaptiveTraining(1,(10,10,1,1))

prob = Sophon.discretize(pde_system, pinn, sampler, strategy)
@time res = Optimization.solve(prob, LBFGS(); maxiters=2000)
```

```@example Schrödinger
phi = pinn.phi
ps = res.u

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in pde_system.domain]

u = [sum(phi.u(([x,t]), ps.u)) for x in xs, t in ts]
v = [sum(phi.v(([x,t]), ps.v)) for x in xs, t in ts]
ψ= [sqrt(first(phi.u(([x,t]), ps.u))^2+first(phi.v(([x,t]), ps.v))^2) for x in xs, t in ts]

axis = (xlabel="x", ylabel="t", title="u")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, u, axis=axis)
ax2, hm2= CairoMakie.heatmap(fig[1, end+1], xs, ts, v, axis= merge(axis, (; title="v")))
fig
```

```@example Schrödinger
axis = (xlabel="x", ylabel="t", title="ψ")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, ψ, axis=axis)
Colorbar(fig[:, end+1], hm1)
fig
```
