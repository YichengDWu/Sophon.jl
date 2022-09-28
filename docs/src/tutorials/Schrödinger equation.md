# Schrödinger equation

$\mathrm{i} \partial_t \psi=-\frac{1}{2} \sigma \partial_{x x} \psi-\beta|\psi|^2 \psi$

Letting $\sigma=\beta=1, \psi=u+v i$,Schrödinger equation can be transformed to the following 2 equations:

$u_t+\frac{1}{2} v_{x x}+\left(u^2+v^2\right) v=0$

$v_t-\frac{1}{2} u_{x x}-\left(u^2+v^2\right) u=0$



```@example Schrödinger
using ModelingToolkit, IntervalSets, Sophon, CairoMakie
using Optimization, OptimizationOptimJL, OptimizationOptimisers

@parameters x,t
@variables u(..),v(..)
Dt = Differential(t)
Dxx = Differential(x)^2

eqs=[0 ~ Dt(u(x,t))+1/2*Dxx(v(x,t))+((v(x,t))^2+(u(x,t))^2)*v(x,t),
     0 ~ Dt(v(x,t))-1/2*Dxx(u(x,t))-((v(x,t))^2+(u(x,t))^2)*u(x,t)]

bcs = [u(x, 0.0) ~ 2sech(x))]

domains = [t ∈ Interval(-0.0, pi/2),
           x ∈ Interval(-5.0, 5.0)]

@named pde_system = PDESystem(eqs, bcs, domains, [x,t], [u(x,t),v(x,t)])
```

```@example Schrödinger
pinn = PINN(u=FullyConnected(2,1,tanh; hidden_dims=2,num_layers=3),
            v=FullyConnected(2,1,tanh; hidden_dims=2,num_layers=3))
            
sampler = QuasiRandomSampler(500, 100)
strategy = NonAdaptiveTraining(1, 10)

prob = Sophon.discretize(pde_system, pinn, sampler, strategy)
@time res = Optimization.solve(prob, LBFGS(); maxiters=2000)
```

```@example Schrödinger
phi = pinn.phi
ps = res.u

xs, ts= [infimum(d.domain):0.01:supremum(d.domain) for d in pde_system.domain]

u = [sum(phi.u(([x,t]), ps.u)) for x in xs, t in ts]
v = [sum(phi.v(([x,t]), ps.v)) for x in xs, t in ts]

ψ= [first(phi.u(([x,t]), ps.u))^2+first(phi.v(([x,t]), ps.v))^2 for x in xs, t in ts]

axis = (xlabel="x", ylabel="t", title="u")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, u, axis=axis)
ax2, hm2= CairoMakie.heatmap(fig[1, end+1], xs, ts, v, axis= merge(axis, (;title = "v")))
fig
```
![](https://upload-images.jianshu.io/upload_images/17163699-8f3dd890bf850a8e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```@example Schrödinger
axis = (xlabel="x", ylabel="t", title="ψ")
fig, ax1, hm1 = CairoMakie.heatmap(xs, ts, ψ, axis=axis)
Colorbar(fig[:, end+1], hm1)
fig
```
![](https://upload-images.jianshu.io/upload_images/17163699-f3b6c02988a6abd0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
