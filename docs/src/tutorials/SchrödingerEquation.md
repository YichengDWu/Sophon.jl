# Schrödinger equation

The nonlinear Shrödinger equation is given by

$\mathrm{i} \partial_t \psi=-\frac{1}{2} \sigma \partial_{x x} \psi-\beta|\psi|^2 \psi$

Let $\sigma=\beta=1, \psi=u+v i$, the equation can be transformed into a system of partial differential equations

$u_t+\frac{1}{2} v_{x x}+\left(u^2+v^2\right) v=0$

$v_t-\frac{1}{2} u_{x x}-\left(u^2+v^2\right) u=0$

```@example Schrödinger
using ModelingToolkit, IntervalSets, Sophon, CairoMakie
using Optimization, OptimizationOptimJL

@parameters x,t
@variables u(..), v(..)
Dₜ = Differential(t)
Dₓ² = Differential(x)^2

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
pinn = PINN(u= Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0),
            v= Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0))
            
sampler = QuasiRandomSampler(2000, (500,500,20,20))
strategy = NonAdaptiveTraining(1,(10,10,1,1))

prob = Sophon.discretize(pde_system, pinn, sampler, strategy)
```
Now we train the neural nets and resample data while training.

```@example Schrödinger
function train(pde_system, prob, sampler, strategy, resample_period = 500, n=10)
     bfgs = BFGS()
     res = Optimization.solve(prob, bfgs; maxiters=2000)
     
     for i in 1:n
         data = Sophon.sample(pde_system, sampler, strategy)
         prob = remake(prob; u0=res.u, p=data)
         res = Optimization.solve(prob, bfgs; maxiters=resample_period)
     end
     return res
end

res = train(pde_system, prob, sampler, strategy)
```

```@example Schrödinger
phi = pinn.phi
ps = res.u

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in pde_system.domain]

u = [sum(phi.u(([x,t]), ps.u)) for x in xs, t in ts]
v = [sum(phi.v(([x,t]), ps.v)) for x in xs, t in ts]
ψ = @. sqrt(u^2+ v^2)

axis = (xlabel="t", ylabel="x", title="u")
fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, u', axis=axis)
ax2, hm2= CairoMakie.heatmap(fig[1, end+1], ts, xs, v', axis= merge(axis, (; title="v")))
display(fig)
save("uv.png", fig); nothing # hide
```
![](uv.png)

```@example Schrödinger
axis = (xlabel="t", ylabel="x", title="ψ")
fig, ax1, hm1 = CairoMakie.heatmap(ts, xs, ψ', axis=axis)
Colorbar(fig[:, end+1], hm1)
display(fig)
save("phi.png", fig); nothing # hide
```
![](phi.png)

## Customize Sampling 

Bascially any sampling method is supportted.

```@example Schrödinger
using StatsBase

data = vec([[x, t] for x in xs, t in ts])
wv = vec(ψ)
new_data = wsample(data, wv, 2000)
new_data = reduce(hcat, new_data)
fig, ax = scatter(new_data[2,:], new_data[1,:])
save("data.png", fig); nothing # hide
```
![](data.png)

```@example Schrödinger
prob.p[1] = new_data
prob.p[2] = new_data
prob = remake(prob; u0 = res.u)
# res = Optimization.solve(prob, bfgs; maxiters=1000)
```
