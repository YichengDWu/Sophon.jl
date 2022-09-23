# Allen-Cahn equation with Sequential Trainning

In this tutorial we are going to solve the Allen-Cahn equation with periodic boundary condition from ``t=0`` to ``t=1``. The traning process is split into four stages, namely 
``t\in [0,0.25]``, ``t\in [0.25,0.5]``, ``t\in [0.5,0.75]`` and ``t\in [0.75, 1.0]``.

```@example allen
using ModelingToolkit, IntervalSets
using Sophon
using Optimization, OptimizationOptimJL

@parameters t, x
@variables u(..)
Dₓ = Differential(x)
Dₓ² = Differential(x)^2
Dₜ = Differential(t)

eq = Dₜ(u(x, t)) - 0.0001 * Dₓ²(u(x, t)) + 5 * u(x,t)^3 - 5 * u(x,t) ~ 0

domain = [x ∈ -1.0..1.0, t ∈ 0.0..0.25]

bcs = [u(x,0) ~ x^2 * cospi(x),
       u(-1,t) ~ u(1,t)]

@named allen = PDESystem(eq, bcs, domain, [x, t], [u(x, t)])
```

Then we define the neural net, the sampler, and the training strategy.
```@example allen
chain = FullyConnected(2, 1, tanh; hidden_dims=16, num_layers=4)
pinn = PINN(chain)
sampler = QuasiRandomSampler(500, (300, 100))
strategy = NonAdaptiveTraining(1, (50, 1))
prob = Sophon.discretize(allen, pinn, sampler, strategy)
```

We solve the equation sequentially in time.

```@example allen
function train(allen, prob, sampler, strategy)
    @time res = Optimization.solve(prob, LBFGS(); maxiters=2000)

    for tmax in [0.5, 0.75, 1.0]
        allen.domain[2] = t ∈ 0.0..tmax
        data = Sophon.sample(allen, sampler, strategy)
        prob = remake(prob; u0=res.u, p=data)
        @time res = Optimization.solve(prob, LBFGS(); maxiters=2000)
    end
end

train(allen, prob, sampler, strategy)
```

Let's plot the result.
```@example allen
using CairoMakie

phi = pinn.phi
xs, ts = [infimum(d.domain):0.01:supremum(d.domain) for d in allen.domain]
axis = (xlabel="t", ylabel="x", title="Prediction")
u_pred = [sum(pinn.phi([x, t], res.u)) for x in xs, t in ts]
fig, ax, hm = heatmap(ts, xs, u_pred', axis=axis)

save("allen.png", fig); nothing # hide
```
![](allen.png)
