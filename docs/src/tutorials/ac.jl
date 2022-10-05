
using ModelingToolkit, IntervalSets
using Sophon, Lux, CUDA, Zygote
using Optimization, OptimizationOptimJL

@parameters t, x
@variables u(..)
Dₓ = Differential(x)
Dₓ² = Differential(x)^2
Dₜ = Differential(t)

eqs = Dₜ(u(x, t)) - 0.0001 * Dₓ²(u(x, t)) + 5 * u(x,t)^3 - 5 * u(x,t) ~ 0

domain = [x ∈ -1.0..1.0, t ∈ 0.0..0.25]

bcs = [u(x,0) ~ x^2 * cospi(x)]
       #u(-1,t) ~ u(1,t)]

@named allen = PDESystem(eq, bcs, domain, [x, t], [u(x, t)])



chain = FullyConnected(2, 1, tanh; hidden_dims=16, num_layers=4)
pinn = PINN(chain) |> gpu
sampler = QuasiRandomSampler(500, 300)
strategy = NonAdaptiveTraining(1, 50)
prob = Sophon.discretize(allen, pinn, sampler, strategy)

Zygote.gradient(p->prob.f(p, prob.p), prob.u0)



derivative = Sophon.numeric_derivative
additional_loss=Sophon.null_additional_loss
(; eqs, bcs, domain, ps, defaults, indvars, depvars) =  allen
(; phi, init_params) = pinn

default_p = ps == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in ps]

using NeuralPDE
depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(indvars,
                                                                                     depvars)

multioutput = phi isa NamedTuple

if !(eqs isa Vector)
    eqs = [eqs]
end

pde_indvars = NeuralPDE.get_variables(eqs, dict_indvars, dict_depvars)

bc_indvars = NeuralPDE.get_variables(bcs, dict_indvars, dict_depvars)

pde_integration_vars = NeuralPDE.get_integration_variables(eqs, dict_indvars,
                                                           dict_depvars)
bc_integration_vars = NeuralPDE.get_integration_variables(bcs, dict_indvars,
                                                          dict_depvars)

pinnrep = (; eqs, bcs, domain, ps, defaults, default_p, additional_loss, depvars,
           indvars, dict_indvars, dict_depvars, dict_depvar_input, multioutput,
           init_params, phi, derivative, strategy, pde_indvars, bc_indvars,
           pde_integration_vars, bc_integration_vars, fdtype = Float64,
           eq_params=SciMLBase.NullParameters(), param_estim=false)
integral = Sophon.get_numeric_integral(pinnrep)
pinnrep = merge(pinnrep, (; integral))


symbolic_pde_loss_functions = [Sophon.build_symbolic_loss_function(pinnrep, eq;
                                                                bc_indvars=pde_indvar)
                                   for (eq, pde_indvar) in zip(eqs, pde_indvars,
                                                               pde_integration_vars)]

    symbolic_bc_loss_functions = [Sophon.build_symbolic_loss_function(pinnrep, bc;
                                                               bc_indvars=bc_indvar)
                                  for (bc, bc_indvar) in zip(bcs, bc_indvars,
                                                             bc_integration_vars)]

    pinnrep = merge(pinnrep, (; symbolic_pde_loss_functions, symbolic_bc_loss_functions))

    datafree_pde_loss_functions = Tuple([Sophon.build_loss_function(pinnrep, eq, pde_indvar)
                                         for (eq, pde_indvar, integration_indvar) in zip(eqs,
                                                                                         pde_indvars,
                                                                                         pde_integration_vars)])

    datafree_bc_loss_functions = Tuple([Sophon.build_loss_function(pinnrep, bc, bc_indvar)
                                        for (bc, bc_indvar, integration_indvar) in zip(bcs,
                                                                                       bc_indvars,
                                                                                       bc_integration_vars)])


ff = datafree_pde_loss_functions[1]
ff(prob.p[1], prob.u0)
ff(cpu(prob.u0), cpu(prob.p[1]))

Zygote.gradient(p->sum(datafree_pde_loss_functions[1](prob.p[1], p)), prob.u0)[1]

Zygote.gradient(p->sum(datafree_pde_loss_functions[1](cpu(prob.p[1]), p)), cpu(prob.u0))[1]


Zygote.gradient(p->sum(datafree_bc_loss_functions[1](prob.p[2], p)), prob.u0)[1]

Zygote.gradient(p->sum(datafree_bc_loss_functions[1](cpu(prob.p[2]), p)), cpu(prob.u0))[1]


Zygote.gradient(θ -> sum(Sophon.numeric_derivative(phi, uu, prob.p[1], [[0.0001220703125, 0.0], [0.0001220703125, 0.0]], 2, θ)), prob.u0)

Zygote.gradient(θ -> sum(Sophon.numeric_derivative(phi, uu, prob.p[1],  [[0.0, 6.055454452393343e-6]], 1, θ)), prob.u0)[1]
