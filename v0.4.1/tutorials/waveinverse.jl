using ModelingToolkit, Sophon
using Optimization, OptimizationOptimJL

@parameters t x
@variables u(..) c²(..)

Dₜ = Differential(t)
Dₓ² = Differential(x)^2
Dₜ² = Differential(t)^2

eq = [Dₜ²(u(t, x)) ~ Dₓ²(u(t, x)) * c²(t, x)]
domain = [x ∈ (-1, 1), t ∈ (0, 1)]

c_true = 1.54
u_true(t, x) = sin(x) * (sin(c_true * t) + cos(c_true * t))
bc = [u(0.0, x) ~ sin(x)]

@named wave = PDESystem(eq, bc, domain, [t, x], [u(t, x), c²(t, x)])

chain_u = FullyConnected((2, 16, 16, 16, 1), sin)
chain_c = ConstantFunction()

pinn = PINN(; u=chain_u, c²=chain_c)
sampler = QuasiRandomSampler(500, 100)
strategy = NonAdaptiveTraining(1, 10)

t_data = rand(1, 10)
x_data = fill(0.8, 1, 10)
input_data = vcat(t_data, x_data)
u_data = u_true.(t_data, x_data)

function additional_loss(phi, θ)
    return sum(abs2, phi.u(input_data, θ.u) .- u_data)
end

prob = Sophon.discretize(wave, pinn, sampler, strategy; additional_loss=additional_loss)

function callback(p, l)
    println("Loss: $l")
    println("c: ", sqrt(abs(p.c².constant[1]))) / 10
    return false
end

res = Optimization.solve(prob, BFGS(); callback=callback, maxiters=2000)

using CairoMakie

ts = range(0, 1; length=100)
xs = range(-1, 1; length=100)

u_pred = [pinn.phi.u([t, x], res.u.u)[1] for t in ts, x in xs]

fig, ax, hm = heatmap(ts, xs, u_pred; colormap=:jet)

u_true_ = [u_true(t, x) for t in ts, x in xs]

fig, ax, hm = heatmap(ts, xs, u_true_; colormap=:jet)
