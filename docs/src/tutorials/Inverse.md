
# Lorenz System

$\frac{\mathrm{d} x}{\mathrm{~d} t}=\sigma(y-x)$,

$\frac{\mathrm{d} y}{\mathrm{~d} t}=x(\rho-z)-y$,

$\frac{\mathrm{d} z}{\mathrm{~d} t}=x y-\beta z$,


```julia
using ModelingToolkit, Sophon, OrdinaryDiffEq
using Optimization, OptimizationOptimJL
using ModelingToolkit, IntervalSets
```


```julia
@parameters t 
@variables x(..), y(..), z(..), σ(..), β(..), ρ(..)

Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ(t)*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ(t) - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β(t)*z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0,1.0)]
@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t),y(t),z(t),σ(t), ρ(t), β(t)])
```

```julia
function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,1.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob, Tsit5(), dt=0.1)
ts = [infimum(d.domain):0.1:supremum(d.domain) for d in domains][1]
function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us,ts_]
end
data = getData(sol)

(u_ , t_) = data
```




    2-element Vector{Matrix{Float64}}:
     [1.0 1.2714705245515625 … -9.129935274368785 -9.397841485892151; 0.0 2.5502284813556653 … -9.919199673331995 -9.094622603195736; 0.0 0.11415728662964351 … 26.980024057777484 28.558082735246224]
     [0.0 0.1 … 0.9 1.0]




```julia
pinn = PINN(x = FullyConnected((1,16,16,16,1), tanh),
            y = FullyConnected((1,16,16,16,1), tanh),
            z = FullyConnected((1,16,16,16,1), tanh),
            σ = ConstantFunction(),
            ρ = ConstantFunction(),
            β = ConstantFunction())
sampler = QuasiRandomSampler(2000, 500)
strategy = NonAdaptiveTraining(1,0)

t_data = t_
u_data = u_ 
function additional_loss(phi, θ)
    return sum(abs2, vcat(phi.x(t_data, θ.x), phi.y(t_data, θ.y), phi.z(t_data, θ.z)).-u_data)/length(t_data)
end
prob = Sophon.discretize(pde_system, pinn, sampler, strategy, additional_loss=additional_loss)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end
@time res = Optimization.solve(prob, BFGS(), callback = callback, maxiters=1000)
```

```julia
#Parameters of inversion
print(res.u.σ_.constant, res.u.ρ.constant, res.u.β.constant)
```

```julia
phi=pinn.phi
θ = res.u
ts=  [0:0.01:1...;;]
x_pred = phi.x(ts, θ.x)
y_pred = phi.x(ts, θ.y)
z_pred = phi.x(ts, θ.z)
```


```julia
using Plots
Plots.plot(vec(ts), [vec(x_pred),vec(y_pred),vec(z_pred)],  title=["x(t)" "y(t)" "z(t)"])   
```







