"""
    CausalTraining(points; epsilon, bcs_points=points, sampling_alg=LatinHypercubeSample())

## Keyword arguments

  - `epsilon`: How much you respect causality. If `epsilon` is 0, then it falls back to `QuasiRandomTraining`. You can also pass in a `AbstractSchedule`.

## References

[1] Wang S, Sankaran S, Perdikaris P. Respecting causality is all you need for training physics-informed neural networks[J]. arXiv preprint arXiv:2203.07404, 2022.

!!!note
You must write a callback function to set `CausalTraining.reweight = true`.
"""
mutable struct CausalTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    init_points::Int64
    bc_points::Int64
    epsilon::AbstractSchedule
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
    reweight::Bool
    W::AbstractMatrix{Float64}
    bc_loss_weights::Vector
end

function CausalTraining(points; epsilon, bc_loss_weights, init_points=points,
                        bc_points=points, sampling_alg=LatinHypercubeSample())
    epsilon = epsilon isa Real ? Constant(Float64(epsilon)) : epsilon
    return CausalTraining(points, init_points, bc_points, epsilon, sampling_alg, false,
                          Array{Float64}(undef, 1, 1), bc_loss_weights)
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::CausalTraining,
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, dict_indvars, bc_indvars, iteration) = pinnrep

    ϵ = strategy.epsilon(first(iteration)) # Currently not working, there is a bug

    tidx = dict_indvars[:t]
    init_idx = Int[]
    bc_idx = Int[]
    for (i, s) in enumerate(bc_indvars)
        :t ∈ s ? push!(bc_idx, i) : push!(init_idx, i)
    end

    eltypeθ = eltype(flat_init_params)
    device = SciMLBase.parameterless_type(ComponentArrays.getdata(flat_init_params))

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds
    real_bcs_bounds = bcs_bounds[bc_idx]
    init_bounds = bcs_bounds[init_idx]
    datafree_init_loss_function = datafree_bc_loss_function[init_idx]
    real_datafree_bc_loss_function = datafree_bc_loss_function[bc_idx]

    strategy_ = QuasiRandomTraining(strategy.init_points;
                                    sampling_alg=strategy.sampling_alg, resampling=false,
                                    minibatch=1)

    init_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy_)
                           for (_loss, bound) in zip(datafree_init_loss_function,
                                                     init_bounds)]
    init_weights = strategy.bc_loss_weights[init_idx]

    init_loss_functions = [θ -> init_weight * f(θ)
                           for (init_weight, f) in zip(init_weights, init_loss_functions)]

    strategy.W = adapt(device,
                       ones(Float64, length(datafree_pde_loss_function), strategy.points))

    if isempty(real_datafree_bc_loss_function)
        pde_loss_functions = get_pde_loss_function(init_loss_functions,
                                                   datafree_pde_loss_function, pde_bounds,
                                                   tidx, eltypeθ, device, ϵ, strategy)
        return pde_loss_functions, init_loss_functions
    else
        pde_loss_functions, bcs_loss_functions = get_pde_and_bc_loss_function(init_loss_functions,
                                                                              real_datafree_bc_loss_function,
                                                                              datafree_pde_loss_function,
                                                                              real_bcs_bounds,
                                                                              pde_bounds,
                                                                              tidx, eltypeθ,
                                                                              device, ϵ,
                                                                              strategy)

        return pde_loss_functions, vcat(bcs_loss_functions, init_loss_functions)
    end
end

function get_pde_and_bc_loss_function(init_loss_functions, datafree_bc_loss_functions,
                                      datafree_pde_functions, bc_bounds, pde_bounds, tidx,
                                      eltypeθ, device, ϵ, strategy)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    bc_points = strategy.bc_points

    bc_sets = [NeuralPDE.generate_quasi_random_points(bc_points, bound, eltypeθ,
                                                      sampling_alg) for bound in bc_bounds]
    bc_sets = [adapt(device, set) for set in bc_sets]

    pde_sets = NeuralPDE.generate_quasi_random_points(points, pde_bounds, eltypeθ,
                                                      sampling_alg)
    pde_sets = [sortslices(set; dims=2, alg=InsertionSort,
                           lt=(x, y) -> isless(x[tidx], y[tidx])) for set in pde_sets]

    pde_set = [adapt(device, set) for set in pde_sets]

    function get_bc_loss_func(bc_loss_func, bc_set)
        return θ -> begin
            set_ = adapt(device, bc_set)
            abs2.(bc_loss_func(set_, θ))
        end
    end

    bc_loss_functions = [get_bc_loss_func(bc_loss_func, bc_set)
                         for (bc_loss_func, bc_set) in zip(datafree_bc_loss_functions,
                                                           bc_sets)]

    function get_pde_loss_function(datafree_pde_loss_func, pde_set)
        return θ -> begin
            ChainRulesCore.@ignore_derivatives begin if strategy.reweight
                L_init = reduce(+, [loss_func(θ) for loss_func in init_loss_functions])
                L_pde = abs2.(datafree_pde_loss_func(pde_set, θ))

                L = hcat(similar(L_pde, 1, 1) .* L_init, L_pde[:, 1:(end - 1)])
                strategy.W = exp.(-ϵ / points .* cumsum(L; dims=2))
                strategy.reweight = false
            end end
            mean(abs2, strategy.W .* datafree_pde_loss_func(pde_set, θ))
        end
    end

    pde_loss_functions = [get_pde_loss_function(pde_loss_func, pde_set)
                          for (pde_loss_func, pde_set) in zip(datafree_pde_functions,
                                                              pde_sets)]
    reduced_bc_loss_functions = [θ -> mean(loss_func(θ)) for loss_func in bc_loss_functions]

    return pde_loss_functions, reduced_bc_loss_functions
end

function get_pde_loss_function(init_loss_functions, datafree_pde_functions, pde_bounds,
                               tidx, eltypeθ, device, ϵ, strategy)
    sampling_alg = strategy.sampling_alg
    points = strategy.points

    pde_sets = [NeuralPDE.generate_quasi_random_points(points, pde_bound, eltypeθ,
                                                       sampling_alg)
                for pde_bound in pde_bounds]
    pde_sets = [sortslices(set; dims=2, alg=InsertionSort,
                           lt=(x, y) -> isless(x[tidx], y[tidx])) for set in pde_sets]
    pde_sets = [adapt(device, set) for set in pde_sets]

    function get_loss_function(i, datafree_pde_loss_func, pde_set)
        return θ -> begin
            ub = points
            ChainRulesCore.@ignore_derivatives begin if strategy.reweight
                L_init = sum(loss_func(θ) for loss_func in init_loss_functions)
                L_pde = abs2.(datafree_pde_loss_func(pde_set, θ))
                L = hcat(adapt(device, [L_init;;]), L_pde[:, 1:(end - 1)])
                strategy.W[i, :] .= vec(exp.(-ϵ / points .* cumsum(L; dims=2)))
                strategy.reweight = false
                ub = findfirst(Base.Fix2(<, 1e-4), strategy.W[i, :])
                ub = isnothing(ub) ? points : ub
            end end

            mean(strategy.W[i:i, 1:ub] .*
                 abs2.(datafree_pde_loss_func(pde_set[:, 1:ub], θ)))
        end
    end

    pde_loss_functions = [get_loss_function(i, pde_loss_func, pde_set)
                          for (i, (pde_loss_func, pde_set)) in enumerate(zip(datafree_pde_functions,
                                                                             pde_sets))]

    return pde_loss_functions
end
