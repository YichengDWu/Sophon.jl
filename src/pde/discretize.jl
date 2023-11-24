function build_loss_function(pde_system::ModelingToolkit.PDESystem, pinn::PINN,
                             strategy::AbstractTrainingAlg, derivative,
                             derivative_bc, fdtype)
    (; eqs, bcs, domain, ps, defaults, indvars, depvars) = pde_system
    (; phi, init_params) = pinn

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(indvars,
                                                                               depvars)

    multioutput = phi isa NamedTuple

    if !(eqs isa Vector)
        eqs = [eqs]
    end

    eqs = map(ModelingToolkit.expand_derivatives, eqs)

    pinnrep = (; eqs, bcs, domain, ps, defaults, depvars, indvars, dict_indvars,
               dict_depvars, dict_depvar_input, multioutput, init_params, phi, derivative,
               strategy, fdtype)

    datafree_pde_loss_functions = Tuple(build_loss_function(pinnrep, eq, i)
                                        for (i, eq) in enumerate(eqs))

    pinnrep = Lux.@set pinnrep.derivative = derivative_bc
    datafree_bc_loss_functions = Tuple(build_loss_function(pinnrep, bc,
                                                           i +
                                                           length(datafree_pde_loss_functions))
                                       for (i, bc) in enumerate(bcs))

    pde_and_bcs_loss_function = scalarize(strategy, phi, datafree_pde_loss_functions,
                                          datafree_bc_loss_functions)
    return pde_and_bcs_loss_function
end

function build_loss_function(pde_system::PDESystem, pinn::PINN,
                             strategy::AbstractTrainingAlg, derivative,
                             derivative_bc, fdtype)
    (; eqs, bcs, ivs, dvs) = pde_system
    (; phi, init_params) = pinn

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(ivs, dvs)

    multioutput = phi isa NamedTuple

    pinnrep = (; eqs, bcs, depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input,
               multioutput, init_params, phi, derivative, strategy,
               fdtype)

    datafree_pde_loss_functions = Tuple(build_loss_function(pinnrep, first(eq), i)
                                        for (i, eq) in enumerate(eqs))

    pinnrep = Lux.@set pinnrep.derivative = derivative_bc
    datafree_bc_loss_functions = Tuple(build_loss_function(pinnrep, first(bc),
                                                           i +
                                                           length(datafree_pde_loss_functions))
                                       for (i, bc) in enumerate(bcs))

    pde_and_bcs_loss_function = scalarize(strategy, phi, datafree_pde_loss_functions,
                                          datafree_bc_loss_functions)
    return pde_and_bcs_loss_function
end

"""
     discretize(pde_system::PDESystem, pinn::PINN, sampler::PINNSampler,
                    strategy::AbstractTrainingAlg; derivative=finitediff,
                    additional_loss)

Convert the PDESystem into an `OptimizationProblem`. You will have access to each loss function
`Sophon.residual_function_1`, `Sophon.residual_function_2`... after calling this function.
"""
function discretize(pde_system, pinn::PINN, sampler::PINNSampler,
                    strategy::AbstractTrainingAlg;
                    additional_loss=Sophon.null_additional_loss, derivative=finitediff,
                    derivative_bc = derivative, fdtype=Float64,
                    adtype=Optimization.AutoZygote())
    datasets = sample(pde_system, sampler)
    init_params = Lux.fmap(Base.Fix1(broadcast, fdtype), pinn.init_params)
    init_params = _ComponentArray(init_params)

    datasets = map(Base.Fix1(broadcast, fdtype), datasets)
    datasets = init_params isa AbstractGPUComponentVector ?
               map(Base.Fix1(adapt, get_gpu_adaptor()), datasets) : datasets
    pde_and_bcs_loss_function = build_loss_function(pde_system, pinn, strategy,
                                                    derivative, derivative_bc,
                                                    fdtype)

    function full_loss_function(θ, p)
        return pde_and_bcs_loss_function(θ, p) + additional_loss(pinn.phi, θ)
    end
    f = OptimizationFunction(full_loss_function, adtype)
    return Optimization.OptimizationProblem(f, init_params, datasets)
end

function symbolic_discretize(pde_system, pinn::PINN, sampler::PINNSampler,
                             strategy::AbstractTrainingAlg;
                             additional_loss=Sophon.null_additional_loss, derivative=finitediff,
                             derivative_bc = derivative, fdtype=Float64,
                             adtype=Optimization.AutoZygote())
    (; eqs, bcs, domain, ps, defaults, indvars, depvars) = pde_system
    (; phi, init_params) = pinn

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(indvars,
                                                                               depvars)

    multioutput = phi isa NamedTuple

    if !(eqs isa Vector)
        eqs = [eqs]
    end

    eqs = map(ModelingToolkit.expand_derivatives, eqs)

    pinnrep = (; eqs, bcs, domain, ps, defaults, depvars, indvars, dict_indvars,
               dict_depvars, dict_depvar_input, multioutput, init_params, phi, derivative,
               strategy, fdtype)

    pde_loss_function = map(eqs) do eq
        args, body = build_symbolic_loss_function(pinnrep, eq)
        return :($args -> $body)
    end

    pinnrep = Lux.@set pinnrep.derivative = derivative_bc
    bc_loss_function = map(bcs) do bc
        args, body = build_symbolic_loss_function(pinnrep, bc)
        return :($args -> $body)
    end
    return [pde_loss_function; bc_loss_function]
end
