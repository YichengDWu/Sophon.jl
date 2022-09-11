function SciMLBase.symbolic_discretize(pde_system::PDESystem,
                                       discretization::PhysicsInformedNN)
    eqs = pde_system.eqs
    bcs = pde_system.bcs
    chain = discretization.chain

    domains = pde_system.domain
    eq_params = pde_system.ps
    defaults = pde_system.defaults
    default_p = eq_params == SciMLBase.NullParameters() ? nothing :
                [defaults[ep] for ep in eq_params]

    param_estim = discretization.param_estim
    additional_loss = discretization.additional_loss
    adaloss = discretization.adaptive_loss

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(pde_system.indvars,
                                                                               pde_system.depvars)

    multioutput = discretization.multioutput
    init_params = discretization.init_params

    if init_params === nothing
        # Use the initialization of the neural network framework
        # But for Lux, default to Float64
        # For Flux, default to the types matching the values in the neural network
        # This is done because Float64 is almost always better for these applications
        # But with Flux there's already a chosen type from the user

        if chain isa AbstractArray
            if chain[1] isa Flux.Chain
                init_params = map(chain) do x
                    _x = Flux.destructure(x)[1]
                end
            else
                x = map(chain) do x
                    _x = ComponentArrays.ComponentArray(Lux.initialparameters(Random.default_rng(),
                                                                              x))
                    Float64.(_x) # No ComponentArray GPU support
                end
                names = ntuple(i -> depvars[i], length(chain))
                init_params = ComponentArrays.ComponentArray(NamedTuple{names}(i
                                                                               for i in x))
            end
        else
            if chain isa Flux.Chain
                init_params = Flux.destructure(chain)[1]
                init_params = init_params isa Array ? Float64.(init_params) :
                              init_params
            else
                init_params = Float64.(ComponentArrays.ComponentArray(Lux.initialparameters(Random.default_rng(),
                                                                                            chain)))
            end
        end
    else
        init_params = init_params
    end

    if (discretization.phi isa Vector && discretization.phi[1].f isa Optimisers.Restructure) ||
       (!(discretization.phi isa Vector) && discretization.phi.f isa Optimisers.Restructure)
        # Flux.Chain
        flat_init_params = multioutput ? reduce(vcat, init_params) : init_params
        flat_init_params = param_estim == false ? flat_init_params :
                           vcat(flat_init_params,
                                adapt(typeof(flat_init_params), default_p))
    else
        flat_init_params = if init_params isa ComponentArrays.ComponentArray
            init_params
        elseif multioutput
            @assert length(init_params) == length(depvars)
            names = ntuple(i -> depvars[i], length(init_params))
            x = ComponentArrays.ComponentArray(NamedTuple{names}(i for i in init_params))
        else
            ComponentArrays.ComponentArray(init_params)
        end
        flat_init_params = if param_estim == false && multioutput
            ComponentArrays.ComponentArray(; depvar = flat_init_params)
        elseif param_estim == false && !multioutput
            flat_init_params
        else
            ComponentArrays.ComponentArray(; depvar = flat_init_params, p = default_p)
        end
    end

    eltypeθ = eltype(flat_init_params)

    if adaloss === nothing
        adaloss = NonAdaptiveLoss{eltypeθ}()
    end

    phi = discretization.phi

    if (phi isa Vector && phi[1].f isa Lux.AbstractExplicitLayer)
        for ϕ in phi
            ϕ.st = adapt(typeof(flat_init_params), ϕ.st)
        end
    elseif (!(phi isa Vector) && phi.f isa Lux.AbstractExplicitLayer)
        phi.st = adapt(typeof(flat_init_params), phi.st)
    end

    derivative = discretization.derivative
    strategy = discretization.strategy

    logger = discretization.logger
    log_frequency = discretization.log_options.log_frequency
    iteration = discretization.iteration
    self_increment = discretization.self_increment

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

    pde_integration_vars = get_integration_variables(eqs, dict_indvars, dict_depvars)
    bc_integration_vars = get_integration_variables(bcs, dict_indvars, dict_depvars)

    pinnrep = PINNRepresentation(eqs, bcs, domains, eq_params, defaults, default_p,
                                 param_estim, additional_loss, adaloss, depvars, indvars,
                                 dict_indvars, dict_depvars, dict_depvar_input, logger,
                                 multioutput, iteration, init_params, flat_init_params, phi,
                                 derivative,
                                 strategy, pde_indvars, bc_indvars, pde_integration_vars,
                                 bc_integration_vars, nothing, nothing, nothing, nothing)

    integral = get_numeric_integral(pinnrep)

    symbolic_pde_loss_functions = [build_symbolic_loss_function(pinnrep, eq;
                                                                bc_indvars = pde_indvar)
                                   for (eq, pde_indvar) in zip(eqs, pde_indvars,
                                                               pde_integration_vars)]

    symbolic_bc_loss_functions = [build_symbolic_loss_function(pinnrep, bc;
                                                               bc_indvars = bc_indvar)
                                  for (bc, bc_indvar) in zip(bcs, bc_indvars,
                                                             bc_integration_vars)]

    pinnrep.integral = integral
    pinnrep.symbolic_pde_loss_functions = symbolic_pde_loss_functions
    pinnrep.symbolic_bc_loss_functions = symbolic_bc_loss_functions

    datafree_pde_loss_functions = [build_loss_function(pinnrep, eq, pde_indvar)
                                   for (eq, pde_indvar, integration_indvar) in zip(eqs,
                                                                                   pde_indvars,
                                                                                   pde_integration_vars)]

    datafree_bc_loss_functions = [build_loss_function(pinnrep, bc, bc_indvar)
                                  for (bc, bc_indvar, integration_indvar) in zip(bcs,
                                                                                 bc_indvars,
                                                                                 bc_integration_vars)]

    pde_loss_functions, bc_loss_functions = merge_strategy_with_loss_function(pinnrep,
                                                                              strategy,
                                                                              datafree_pde_loss_functions,
                                                                              datafree_bc_loss_functions)

    if adaloss isa AugmentedLagrangian
        all_loss_function = generate_full_loss_function(pinnrep, adaloss,
                                                        pde_loss_functions,
                                                        bc_loss_functions,
                                                        additional_loss)
        pinnrep.loss_functions = PINNLossFunctions(bc_loss_functions, pde_loss_functions,
                                                   all_loss_function, additional_loss,
                                                   datafree_pde_loss_functions,
                                                   datafree_bc_loss_functions)
    else
        # setup for all adaptive losses
        num_pde_losses = length(pde_loss_functions)
        num_bc_losses = length(bc_loss_functions)
        # assume one single additional loss function if there is one. this means that the user needs to lump all their functions into a single one,
        num_additional_loss = additional_loss isa Nothing ? 0 : 1

        adaloss_T = eltype(adaloss.pde_loss_weights)

        # this will error if the user has provided a number of initial weights that is more than 1 and doesn't match the number of loss functions
        adaloss.pde_loss_weights = ones(adaloss_T, num_pde_losses) .* adaloss.pde_loss_weights
        adaloss.bc_loss_weights = ones(adaloss_T, num_bc_losses) .* adaloss.bc_loss_weights
        adaloss.additional_loss_weights = ones(adaloss_T, num_additional_loss) .*
                                        adaloss.additional_loss_weights

        reweight_losses_func = generate_adaptive_loss_function(pinnrep, adaloss,
                                                            pde_loss_functions,
                                                            bc_loss_functions)

        function full_loss_function(θ, p)
            # the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
            pde_losses = [pde_loss_function(θ) for pde_loss_function in pde_loss_functions]
            bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

            # this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
            # that's why we prefer the user to maintain the increment in the outer loop callback during optimization
            ChainRulesCore.@ignore_derivatives if self_increment
                iteration[1] += 1
            end

            ChainRulesCore.@ignore_derivatives begin reweight_losses_func(θ, pde_losses,
                                                                        bc_losses) end

            weighted_pde_losses = adaloss.pde_loss_weights .* pde_losses
            weighted_bc_losses = adaloss.bc_loss_weights .* bc_losses

            sum_weighted_pde_losses = sum(weighted_pde_losses)
            sum_weighted_bc_losses = sum(weighted_bc_losses)
            weighted_loss_before_additional = sum_weighted_pde_losses + sum_weighted_bc_losses

            full_weighted_loss = if additional_loss isa Nothing
                weighted_loss_before_additional
            else
                function _additional_loss(phi, θ)
                    (θ_, p_) = if (param_estim == true)
                        if (phi isa Vector && phi[1].f isa Optimisers.Restructure) ||
                        (!(phi isa Vector) && phi.f isa Optimisers.Restructure)
                            # Isa Flux Chain
                            θ[1:(end - length(default_p))], θ[(end - length(default_p) + 1):end]
                        else
                            θ.depvar, θ.p
                        end
                    else
                        θ, nothing
                    end
                    return additional_loss(phi, θ_, p_)
                end
                weighted_additional_loss_val = adaloss.additional_loss_weights[1] *
                                            _additional_loss(phi, θ)
                weighted_loss_before_additional + weighted_additional_loss_val
            end

            ChainRulesCore.@ignore_derivatives begin if iteration[1] % log_frequency == 0
                logvector(pinnrep.logger, pde_losses, "unweighted_loss/pde_losses",
                        iteration[1])
                logvector(pinnrep.logger, bc_losses, "unweighted_loss/bc_losses", iteration[1])
                logvector(pinnrep.logger, weighted_pde_losses,
                        "weighted_loss/weighted_pde_losses",
                        iteration[1])
                logvector(pinnrep.logger, weighted_bc_losses,
                        "weighted_loss/weighted_bc_losses",
                        iteration[1])
                logscalar(pinnrep.logger, sum_weighted_pde_losses,
                        "weighted_loss/sum_weighted_pde_losses", iteration[1])
                logscalar(pinnrep.logger, sum_weighted_bc_losses,
                        "weighted_loss/sum_weighted_bc_losses", iteration[1])
                logscalar(pinnrep.logger, full_weighted_loss,
                        "weighted_loss/full_weighted_loss",
                        iteration[1])
                logvector(pinnrep.logger, adaloss.pde_loss_weights,
                        "adaptive_loss/pde_loss_weights",
                        iteration[1])
                logvector(pinnrep.logger, adaloss.bc_loss_weights,
                        "adaptive_loss/bc_loss_weights",
                        iteration[1])
            end end

            return full_weighted_loss
        end

        pinnrep.loss_functions = PINNLossFunctions(bc_loss_functions, pde_loss_functions,
                                                full_loss_function, additional_loss,
                                                datafree_pde_loss_functions,
                                                datafree_bc_loss_functions)
    end

    return pinnrep
end
