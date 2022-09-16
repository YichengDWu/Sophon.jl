function SciMLBase.symbolic_discretize(pde_system::PDESystem, discretization::PINN)
    (; eqs, bcs, domain, ps, defaults, indvars, depvars) = pde_system
    (; phi, init_params, strategy, derivative, additional_loss, adaptive_loss, kwargs) = discretization

    default_p = ps == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in ps]

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(indvars,
                                                                                         depvars)

    multioutput = phi isa NamedTuple

    if isnothing(init_params)
        init_params = Lux.initialparameters(Random.default_rng(), phi) |>
                      ComponentArray .|>
                      Float64
    else
        init_params = init_params
    end

    eltypeθ = eltype(init_params)

    if phi isa NamedTuple
        map(phi) do ϕ
            Lux.@set! ϕ.state = adapt(parameterless_type(getdata(init_params)), ϕ.state)
        end
    else
        Lux.@set! phi.state = adapt(parameterless_type(getdata(init_params)), phi.state)
    end

    if !(eqs isa Array)
        eqs = [eqs]
    end

    pde_indvars = if strategy isa QuadratureTraining
        get_argument(eqs, dict_indvars, dict_depvars)
    else
        get_variables(eqs, dict_indvars, dict_depvars)
    end

    bc_indvars = if strategy isa QuadratureTraining
        get_argument(bcs, dict_indvars, dict_depvars)
    else
        get_variables(bcs, dict_indvars, dict_depvars)
    end

    pde_integration_vars = NeuralPDE.get_integration_variables(eqs, dict_indvars,
                                                               dict_depvars)
    bc_integration_vars = NeuralPDE.get_integration_variables(bcs, dict_indvars,
                                                              dict_depvars)

    pinnrep = NeuralPDE.PINNRepresentation(eqs, bcs, domain, ps, defaults, default_p, false,
                                           additional_loss, adaptive_loss, depvars, indvars,
                                           dict_indvars, dict_depvars, dict_depvar_input,
                                           nothing, multioutput, [1], init_params,
                                           init_params, phi, derivative, strategy,
                                           pde_indvars, bc_indvars, pde_integration_vars,
                                           bc_integration_vars, nothing, nothing, nothing,
                                           nothing)

    integral = NeuralPDE.get_numeric_integral(pinnrep)

    symbolic_pde_loss_functions = [NeuralPDE.build_symbolic_loss_function(pinnrep, eq;
                                                                          bc_indvars=pde_indvar)
                                   for (eq, pde_indvar) in zip(eqs, pde_indvars,
                                                               pde_integration_vars)]

    symbolic_bc_loss_functions = [NeuralPDE.build_symbolic_loss_function(pinnrep, bc;
                                                                         bc_indvars=bc_indvar)
                                  for (bc, bc_indvar) in zip(bcs, bc_indvars,
                                                             bc_integration_vars)]

    pinnrep.integral = integral
    pinnrep.symbolic_pde_loss_functions = symbolic_pde_loss_functions
    pinnrep.symbolic_bc_loss_functions = symbolic_bc_loss_functions

    datafree_pde_loss_functions = [NeuralPDE.build_loss_function(pinnrep, eq, pde_indvar)
                                   for (eq, pde_indvar, integration_indvar) in zip(eqs,
                                                                                   pde_indvars,
                                                                                   pde_integration_vars)]

    datafree_bc_loss_functions = [NeuralPDE.build_loss_function(pinnrep, bc, bc_indvar)
                                  for (bc, bc_indvar, integration_indvar) in zip(bcs,
                                                                                 bc_indvars,
                                                                                 bc_integration_vars)]

    pde_loss_functions, bc_loss_functions = NeuralPDE.merge_strategy_with_loss_function(pinnrep,
                                                                                        strategy,
                                                                                        datafree_pde_loss_functions,
                                                                                        datafree_bc_loss_functions)

    # setup for all adaptive losses
    num_pde_losses = length(pde_loss_functions)
    num_bc_losses = length(bc_loss_functions)
    # assume one single additional loss function if there is one. this means that the user needs to lump all their functions into a single one,
    num_additional_loss = additional_loss isa Nothing ? 0 : 1

    adaptive_loss_T = eltype(adaptive_loss.pde_loss_weights)

    # this will error if the user has provided a number of initial weights that is more than 1 and doesn't match the number of loss functions
    adaptive_loss.pde_loss_weights = ones(adaptive_loss_T, num_pde_losses) .*
                                     adaptive_loss.pde_loss_weights
    adaptive_loss.bc_loss_weights = ones(adaptive_loss_T, num_bc_losses) .*
                                    adaptive_loss.bc_loss_weights
    adaptive_loss.additional_loss_weights = ones(adaptive_loss_T, num_additional_loss) .*
                                            adaptive_loss.additional_loss_weights

    reweight_losses_func = NeuralPDE.generate_adaptive_loss_function(pinnrep, adaptive_loss,
                                                                     pde_loss_functions,
                                                                     bc_loss_functions)

    function full_loss_function(θ, p)

        # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
        pde_losses = [pde_loss_function(θ) for pde_loss_function in pde_loss_functions]
        bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

        ChainRulesCore.@ignore_derivatives begin reweight_losses_func(θ, pde_losses,
                                                                      bc_losses) end

        weighted_pde_losses = adaptive_loss.pde_loss_weights .* pde_losses
        weighted_bc_losses = adaptive_loss.bc_loss_weights .* bc_losses

        sum_weighted_pde_losses = sum(weighted_pde_losses)
        sum_weighted_bc_losses = sum(weighted_bc_losses)
        weighted_loss_before_additional = sum_weighted_pde_losses + sum_weighted_bc_losses

        full_weighted_loss = if additional_loss isa Nothing
            weighted_loss_before_additional
        else
            function _additional_loss(phi, θ)
                return additional_loss(phi, θ, nothing)
            end
            weighted_additional_loss_val = adaptive_loss.additional_loss_weights[1] *
                                           _additional_loss(phi, θ)
            weighted_loss_before_additional + weighted_additional_loss_val
        end

        return full_weighted_loss
    end

    pinnrep.loss_functions = NeuralPDE.PINNLossFunctions(bc_loss_functions,
                                                         pde_loss_functions,
                                                         full_loss_function,
                                                         additional_loss,
                                                         datafree_pde_loss_functions,
                                                         datafree_bc_loss_functions)

    return pinnrep
end

function SciMLBase.discretize(pde_system::PDESystem, discretization::PINN)
    pinnrep = symbolic_discretize(pde_system, discretization)
    f = OptimizationFunction(pinnrep.loss_functions.full_loss_function,
                             Optimization.AutoZygote())
    return Optimization.OptimizationProblem(f, pinnrep.flat_init_params)
end
