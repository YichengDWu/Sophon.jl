"""

## References

[1] Wu, Chenxi, et al. "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks." arXiv preprint arXiv:2207.10289 (2022).
"""
struct RADTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    bcs_points::Int64
    k::Float64
    c::Float64
    resample_at::Int64
    training_strategy::QuasiRandomTraining
    refinement::Bool
end

function RADTraining(points; resample_at=1, k=1.0, c=k / 100,
                     sampling_alg=LatinHypercubeSample(), bcs_points=points,
                     refinement=true)
    training_strategy = QuasiRandomTraining(points; bcs_points=bcs_points,
                                            sampling_alg=sampling_alg)
    return RADTraining(points, bcs_points, k, c, resample_at, training_strategy, refinement)
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::RADTraining,
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, iteration) = pinnrep

    eltypeθ = eltype(ComponentArrays.getdata(pinnrep.flat_init_params))

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    if iteration[1] <= strategy.resample_at
        return NeuralPDE.merge_strategy_with_loss_function(pinnrep,
                                                           strategy.training_strategy,
                                                           datafree_pde_loss_function,
                                                           datafree_bc_loss_function)

    else
        pde_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy)
                              for (_loss, bound) in zip(datafree_pde_loss_function,
                                                        pde_bounds)]

        bc_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy)
                             for (_loss, bound) in zip(datafree_bc_loss_function,
                                                       bcs_bounds)]

        pde_loss_functions, bc_loss_functions
        return pde_loss_functions, bc_loss_functions
    end
end

function NeuralPDE.get_loss_function(loss_function, bound, eltypeθ, strategy::RADTraining)
    sampling_alg = strategy.training_strategy.sampling_alg
    points = strategy.points
    k = strategy.k
    c = strategy.c

    loss = θ -> begin
        set = NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
        subset = residual_based_sample(loss_function, set, θ, points, k, c)
        dataset = strategy.refinement ? hcat(subset, set) : subset
        mean(abs2, loss_function(dataset, θ))
    end
    return loss
end

ChainRulesCore.@non_differentiable function residual_based_sample(loss_function, set, θ, n,
                                                                  k=2.0, c=k / 100)
    ϵᵏ = (loss_function(set, θ)) .^ k
    w = vec(ϵᵏ .+ c * mean(ϵᵏ))
    subset = wsample([p for p in eachcol(sets)], w, n)
    subset = reduce(hcat, subset)
    subset = adapt(SciMLBase.parameterless_type(ComponentArrays.getdata(θ)), subset)
    return subset
end
