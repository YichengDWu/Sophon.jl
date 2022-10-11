using ModelingToolkit, Sophon
using Optimization, OptimizationOptimJL
using ModelingToolkit, IntervalSets
using QuasiMonteCarlo

@parameters t
@variables x(..), y(..), z(..)

σ = 10.0
ρ = 28.0
β = 8/3

Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β*z(t)]

bcs = [x(0) ~ y(0)] # dummy boundary condition

tmin = 0.0
tmax = 0.1
domains = [t ∈ Interval(tmin, tmax)]
@named ode_system = PDESystem(eqs, bcs, domains, [t], [x(t),y(t),z(t)])

sampler = QuasiRandomSampler(500,1)
pinn = PINN(x = FullyConnected((1,16,16,16,1), tanh),
            y = FullyConnected((1,16,16,16,1), tanh),
            z = FullyConnected((1,16,16,16,1), tanh))

strategy = NonAdaptiveTraining(1,0)

t0 = [0.0;;]
u0 = [1.0 0.0 0.0]

function get_init_loss(t0, u0)
    function init_loss(phi, θ)
        x0 = phi.x(t0, θ.x)
        y0 = phi.y(t0, θ.y)
        z0 = phi.z(t0, θ.z)
        u0_pred = hcat(x0, y0, z0)
        return sum(abs2, u0_pred .- u0) * 10
    end
    return init_loss
end


prob = Sophon.discretize(ode_system, pinn, sampler, strategy; additional_loss = get_init_loss(t0, u0))


function callback(p, l)
    println("Loss: $l")
    if l < 1e-14
        return true
    else
        return false
    end
end



res = Optimization.solve(prob, BFGS(); maxiters = 2000, callback = callback)

dt = 0.5

θs = [tmin => res.u]

θs = Dict(θs...)

function get_u0(phi, θ, t0)
    x0 = phi.x(t0, θ.x)
    y0 = phi.y(t0, θ.y)
    z0 = phi.z(t0, θ.z)
    u0 = hcat(x0, y0, z0)
    return u0
end

function ode_solve(ode_system, pinn, sampler, strategy, θs, tmin, tmax, u0, maxiters)
    tmax > 20 && return nothing
    println("Start training on ($tmin, $tmax)")
    ode_system.domain[1] = t ∈ Interval(tmin, tmax)
    data = Sophon.sample(ode_system, sampler, strategy)
    prob = Sophon.discretize(ode_system, pinn, sampler, strategy; additional_loss = get_init_loss([tmin;;], u0))
    prob = remake(prob; u0 = θs[tmin],  p = data)
    res = Optimization.solve(prob, BFGS(); maxiters = maxiters, callback = callback)
    #if res.minimum > 1e-6
   #     println("Failed to converge at t=$tmax. Trying to retrain the model.")
    tmin = tmin + dt
    tmax = tmax + dt
    θs[tmin] = res.u

    u0 = get_u0(pinn.phi, res.u, [tmin;;])
    return ode_solve(ode_system, pinn, sampler, strategy, θs, tmin, tmax, u0, maxiters)
  #  else
   #     θs[tmin] = res.u
  #      return ode_solve(ode_system, pinn, sampler, strategy, θs, tmin, tmax+dt, u0, maxiters)
 #   end
end

ode_solve(ode_system, pinn, sampler, strategy, θs, tmin, dt, u0, 2000)


phi=pinn.phi

fig = plot()
for (tmin, θ) in θs
    ts = [tmin:0.01:tmin+dt...;;]
    x_pred = phi.x(ts, θ.x)
    y_pred = phi.x(ts, θ.y)
    z_pred = phi.x(ts, θ.z)
    Plots.plot!(vec(ts), [vec(x_pred),vec(y_pred),vec(z_pred)], label = "")
end

fig

θ = θs[0.0]
ts=  [0:0.01:0.5...;;]
x_pred = phi.x(ts, θ.x)
y_pred = phi.x(ts, θ.y)
z_pred = phi.x(ts, θ.z)

using Plots
Plots.plot(vec(ts), [vec(x_pred),vec(y_pred),vec(z_pred)],  label=["x(t)" "y(t)" "z(t)"])
