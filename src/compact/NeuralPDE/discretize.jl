
function get_datafree_pinn_loss_function(pde_system::PDESystem, pinn::PINN,
                                         strategy::AbstractTrainingAlg;
                                         additional_loss=Sophon.null_additional_loss,
                                         derivative=NeuralPDE.numeric_derivative)
    (; eqs, bcs, domain, ps, defaults, indvars, depvars) = pde_system
    (; phi, init_params, kwargs) = pinn
    (; pde_weights, bcs_weights) = strategy

    default_p = ps == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in ps]

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
               pde_integration_vars, bc_integration_vars,
               eq_params=SciMLBase.NullParameters(), param_estim=false)
    integral = get_numeric_integral(pinnrep)
    pinnrep = merge(pinnrep, (; integral))

    symbolic_pde_loss_functions = [build_symbolic_loss_function(pinnrep, eq;
                                                                bc_indvars=pde_indvar)
                                   for (eq, pde_indvar) in zip(eqs, pde_indvars,
                                                               pde_integration_vars)]

    symbolic_bc_loss_functions = [build_symbolic_loss_function(pinnrep, bc;
                                                               bc_indvars=bc_indvar)
                                  for (bc, bc_indvar) in zip(bcs, bc_indvars,
                                                             bc_integration_vars)]

    pinnrep = merge(pinnrep, (; symbolic_pde_loss_functions, symbolic_bc_loss_functions))

    datafree_pde_loss_functions = Tuple([build_loss_function(pinnrep, eq, pde_indvar)
                                         for (eq, pde_indvar, integration_indvar) in zip(eqs,
                                                                                         pde_indvars,
                                                                                         pde_integration_vars)])

    datafree_bc_loss_functions = Tuple([build_loss_function(pinnrep, bc, bc_indvar)
                                        for (bc, bc_indvar, integration_indvar) in zip(bcs,
                                                                                       bc_indvars,
                                                                                       bc_integration_vars)])

    pde_and_bcs_loss_function = get_pde_and_bcs_loss_function(strategy,
                                                              datafree_pde_loss_functions,
                                                              datafree_bc_loss_functions)

    function full_loss_function(θ, p)
        return pde_and_bcs_loss_function(θ, p) + additional_loss(phi, θ)
    end
    return full_loss_function
end

function discretize(pde_system::PDESystem, pinn::PINN{T}, sampler::PINNSampler{S},
                    strategy::AbstractTrainingAlg;
                    additional_loss=Sophon.null_additional_loss,
                    derivative=NeuralPDE.numeric_derivative) where {T, S}
    if parameterless_type(S) != parameterless_type(T)
        throw(ArgumentError("The device type of the sampler and the type of the neural network must be the same. Got $(S) and $(T)"))
    end
    pde_datasets, boundary_datasets = sample(sampler)
    loss_function = get_datafree_pinn_loss_function(pde_system, pinn, strategy;
                                                    additional_loss=additional_loss,
                                                    derivative=derivative)
    f = OptimizationFunction(loss_function, Optimization.AutoZygote())
    return Optimization.OptimizationProblem(f, pinn.init_params, [pde_datasets; boundary_datasets])
end
