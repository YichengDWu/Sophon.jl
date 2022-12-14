
function build_loss_function(pde_system::NeuralPDE.PDESystem, pinn::PINN,
                             strategy::AbstractTrainingAlg; derivative=finitediff)
    (; eqs, bcs, domain, ps, defaults, indvars, depvars) = pde_system
    (; phi, init_params) = pinn

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

    pinnrep = (; eqs, bcs, domain, ps, defaults, default_p, depvars, indvars, dict_indvars,
               dict_depvars, dict_depvar_input, multioutput, init_params, phi, derivative,
               strategy, pde_indvars, bc_indvars, pde_integration_vars, bc_integration_vars,
               fdtype=Float64, eq_params=SciMLBase.NullParameters())
    integral = get_numeric_integral(pinnrep)
    pinnrep = merge(pinnrep, (; integral))

    datafree_pde_loss_functions = Tuple(build_loss_function(pinnrep, eq, i)
                                        for (i, (eq, integration_indvar)) in enumerate(zip(eqs,
                                                                                           pde_integration_vars)))

    datafree_bc_loss_functions = Tuple(build_loss_function(pinnrep, bc,
                                                           i +
                                                           length(datafree_pde_loss_functions))
                                       for (i, (bc, integration_indvar)) in enumerate(zip(bcs,
                                                                                          bc_integration_vars)))

    pde_and_bcs_loss_function = scalarize(strategy, phi, datafree_pde_loss_functions,
                                          datafree_bc_loss_functions)
    return pde_and_bcs_loss_function
end

function build_loss_function(pde_system::PDESystem, pinn::PINN,
                             strategy::AbstractTrainingAlg; derivative=finitediff)
    (; eqs, bcs, ivs, dvs) = pde_system
    (; phi, init_params) = pinn

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(ivs,
                                                                                         dvs)

    multioutput = phi isa NamedTuple

    pde_indvars = NeuralPDE.get_variables(map(first, eqs), dict_indvars, dict_depvars)

    bc_indvars = NeuralPDE.get_variables(map(first, bcs), dict_indvars, dict_depvars)

    pde_integration_vars = NeuralPDE.get_integration_variables(map(first, eqs),
                                                               dict_indvars, dict_depvars)
    bc_integration_vars = NeuralPDE.get_integration_variables(map(first, bcs), dict_indvars,
                                                              dict_depvars)

    pinnrep = (; eqs, bcs, depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input,
               multioutput, init_params, phi, derivative, strategy, pde_indvars, bc_indvars,
               pde_integration_vars, bc_integration_vars, fdtype=Float64,
               eq_params=SciMLBase.NullParameters())
    integral = nothing
    pinnrep = merge(pinnrep, (; integral))

    datafree_pde_loss_functions = Tuple(build_loss_function(pinnrep, first(eq), i)
                                        for (i, (eq, integration_indvar)) in enumerate(zip(eqs,
                                                                                           pde_integration_vars)))

    datafree_bc_loss_functions = Tuple(build_loss_function(pinnrep, first(bc),
                                                           i +
                                                           length(datafree_pde_loss_functions))
                                       for (i, (bc, integration_indvar)) in enumerate(zip(bcs,
                                                                                          bc_integration_vars)))

    pde_and_bcs_loss_function = scalarize(strategy, phi, datafree_pde_loss_functions,
                                          datafree_bc_loss_functions)
    return pde_and_bcs_loss_function
end

function build_loss_function(pde_system::ParametricPDESystem, pinn::PINN,
                             strategy::AbstractTrainingAlg, cord_branch_net;
                             derivative=finitediff)
    (; eqs, bcs, ivs, dvs, pvs) = pde_system
    (; phi, init_params) = pinn

    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(ivs,
                                                                                         dvs)
    _, _, _, dict_pmdepvars, dict_pmdepvar_input = NeuralPDE.get_vars(ivs, pvs)

    multioutput = false

    pde_indvars = NeuralPDE.get_variables(map(first, eqs), dict_indvars, dict_depvars)
    bc_indvars = pde_indvars

    pde_integration_vars = NeuralPDE.get_integration_variables(map(first, eqs),
                                                               dict_indvars, dict_depvars)
    bc_integration_vars = NeuralPDE.get_integration_variables(map(first, bcs), dict_indvars,
                                                              dict_depvars)

    pinnrep = (; eqs, bcs, depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input,
               dict_pmdepvars, dict_pmdepvar_input, multioutput, pvs, init_params, pinn,
               derivative, strategy, pde_indvars, bc_indvars, pde_integration_vars,
               bc_integration_vars, fdtype=Float64, cord_branch_net,
               eq_params=SciMLBase.NullParameters())

    datafree_pde_loss_functions = Tuple(build_loss_function(pinnrep, first(eq), i)
                                        for (i, (eq, integration_indvar)) in enumerate(zip(eqs,
                                                                                           pde_integration_vars)))

    datafree_bc_loss_functions = Tuple(build_loss_function(pinnrep, first(bc),
                                                           i +
                                                           length(datafree_pde_loss_functions))
                                       for (i, (bc, integration_indvar)) in enumerate(zip(bcs,
                                                                                          bc_integration_vars)))

    pde_and_bcs_loss_function = scalarize(strategy, phi, datafree_pde_loss_functions,
                                          datafree_bc_loss_functions)
    return pde_and_bcs_loss_function
end
"""
     discretize(pde_system::PDESystem, pinn::PINN, sampler::PINNSampler,
                    strategy::AbstractTrainingAlg;
                    additional_loss)

Convert the PDESystem into an `OptimizationProblem`. You will have access to each loss function
`Sophon.residual_function_1`, `Sophon.residual_function_2`... after calling this function.
"""
function discretize(pde_system, pinn::PINN, sampler::PINNSampler,
                    strategy::AbstractTrainingAlg;
                    additional_loss=Sophon.null_additional_loss, derivative=finitediff,
                    adtype=Optimization.AutoZygote())
    datasets = sample(pde_system, sampler, strategy)
    init_params = _ComponentArray(pinn.init_params)
    datasets = init_params isa AbstractGPUComponentVector ?
               map(Base.Fix1(adapt, CuArray), datasets) : datasets
    pde_and_bcs_loss_function = build_loss_function(pde_system, pinn, strategy;
                                                    derivative=derivative)

    function full_loss_function(??, p)
        return pde_and_bcs_loss_function(??, p) + additional_loss(pinn.phi, ??)
    end
    f = OptimizationFunction(full_loss_function, adtype)
    return Optimization.OptimizationProblem(f, init_params, datasets)
end

function discretize(pde_system::ParametricPDESystem, pinn::PINN, sampler::PINNSampler,
                    strategy::AbstractTrainingAlg, functionsampler::FunctionSampler,
                    cord_branch_net::AbstractArray;
                    additional_loss=Sophon.null_additional_loss, derivative=finitediff,
                    adtype=Optimization.AutoZygote())
    datasets = sample(pde_system, sampler, strategy)
    init_params = _ComponentArray(pinn.init_params)
    datasets = init_params isa AbstractGPUComponentVector ?
               map(Base.Fix1(adapt, CuArray), datasets) : datasets

    pfs = sample(functionsampler)
    cord_branch_net = cord_branch_net isa Union{AbstractVector, StepRangeLen} ?
                      [cord_branch_net] : cord_branch_net
    pde_and_bcs_loss_function = build_loss_function(pde_system, pinn, strategy,
                                                    cord_branch_net; derivative=derivative)
    function full_loss_function(??, p)
        return pde_and_bcs_loss_function(??, p) + additional_loss(pinn.phi, ??)
    end
    f = OptimizationFunction(full_loss_function, adtype)

    p = PINOParameterHandler(datasets, pfs)
    return Optimization.OptimizationProblem(f, init_params, p)
end
