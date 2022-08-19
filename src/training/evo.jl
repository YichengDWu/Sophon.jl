"""
## References

[1] Wu, Chenxi, et al. "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks." arXiv preprint arXiv:2207.10289 (2022).
"""
struct EvoTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    bcs_points::Int64
    training_strategy::QuasiRandomTraining
end

function EvoTraining(points; sampling_alg=LatinHypercubeSample(), bcs_points=points)
    sampling_strategy = QuasiRandomTraining(points; bcs_points=bcs_points,
                                            sampling_alg=sampling_alg)
    return EvoTrainin(points, bcs_points, sampling_strategy)
end

mutable struct EvoDataset
    set
end

NeuralPDE.@nograd function generate_and_adapt_set(points, bound, eltypeθ, device, sampling_alg)
    set = NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
    set = adapt(device, set)
    return set
end

NeuralPDE.@nograd function updateset!(loss_func, set::EvoDataset, θ, points, bound, eltypeθ, device, sampling_alg)
    losses = abs.(loss_func(set.set, θ))
    epectation = mean(losses)
    set_r = set.set[:, vec(loss) .> epectation]
    set_s = generate_and_adapt_set(points, bound, eltypeθ, device, sampling_alg)
    set_total = hcat(set_r, set_s)
    set.set = set_total
    return set_total
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::RADTraining,
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, iteration) = pinnrep

    eltypeθ = eltype(ComponentArrays.getdata(pinnrep.flat_init_params))
    device = SciMLBase.parameterless_type(ComponentArrays.getdata(flat_init_params))

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    bc_datasets = [EvoDataset(generate_and_adapt_set(strategy.bcs_points, bound, eltypeθ, device,
                              strategy.sampling_alg)) for bound in bcs_bounds]

    pde_datasets = [EvoDataset(generate_and_adapt_set(strategy.points, bound, eltypeθ, device,
                               strategy.sampling_alg)) for bound in pde_bounds]

    pde_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy, set)
                          for (_loss, bound, set) in zip(datafree_pde_loss_function, pde_bounds, pde_datasets)]

    bc_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy, set)
                         for (_loss, bound, set) in zip(datafree_bc_loss_function, bcs_bounds, bc_datasets)]

    pde_loss_functions, bc_loss_functions
    return pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, bound, eltypeθ, strategy::RADTraining, set::EvoDataset)
    sampling_alg = strategy.training_strategy.sampling_alg
    points = strategy.points

    loss = θ -> begin
        device = SciMLBase.parameterless_type(ComponentArrays.getdata(θ))
        new_set = updateset!(loss_function, set, θ, points, bound, eltypeθ, device, sampling_alg)
        mean(abs2, loss_function(new_set, θ))
    end
    return loss
end
