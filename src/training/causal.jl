"""
    CausalTraining(points; epsilon, bcs_points=points, sampling_alg=LatinHypercubeSample())

## Keyword arguments

  - `epsilon`: How much you respect causality. If `epsilon` is 0, then it falls back to `QuasiRandomTraining`. You can also pass in a `AbstractSchedule`.

## References

[1] Wang S, Sankaran S, Perdikaris P. Respecting causality is all you need for training physics-informed neural networks[J]. arXiv preprint arXiv:2203.07404, 2022.
"""
struct CausalTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    init_points::Int64
    epsilon::AbstractSchedule
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
end

function CausalTraining(points; epsilon, init_points=points,
                        sampling_alg=LatinHypercubeSample())
    epsilon = epsilon isa Real ? Constant(Float64(epsilon)) : epsilon
    return CausalTraining(points, init_points, epsilon, sampling_alg)
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

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds
    real_bcs_bounds = bcs_bounds[bc_idx]
    init_bounds = bcs_bounds[init_idx]
    datafree_init_loss_function = datafree_bc_loss_function[init_idx]
    real_datafree_bc_loss_function = datafree_bc_loss_function[bc_idx]

    strategy_ = QuasiRandomTraining(strategy.init_points;
                                    sampling_alg=strategy.sampling_alg)
    init_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy_)
                           for (_loss, bound) in zip(datafree_init_loss_function,
                                                     init_bounds)]

    if isempty(real_datafree_bc_loss_function)
        pde_loss_functions = get_pde_loss_function(init_loss_functions,
                                                   datafree_pde_functions, pde_bounds, tidx,
                                                   eltypeθ, device, ϵ, stategy)
        return pde_loss_functions, init_loss_functions
    else
        pde_loss_functions, bcs_loss_functions = get_pde_and_bc_loss_function(init_loss_functions,
                                                                              real_datafree_bc_loss_function,
                                                                              datafree_pde_loss_function,
                                                                              real_bcs_bounds,
                                                                              pde_bounds,
                                                                              tidx, eltypeθ,
                                                                              SciMLBase.parameterless_type(ComponentArrays.getdata(flat_init_params)),
                                                                              ϵ, strategy)

        return pde_loss_functions, vcat(bcs_loss_functions, init_loss_functions)
    end
end

function get_pde_and_bc_loss_function(init_loss_functions, datafree_bc_loss_functions,
                                      datafree_pde_functions, bc_bounds, pde_bounds, tidx,
                                      eltypeθ, device, ϵ, stategy)
    sampling_alg = stategy.sampling_alg
    points = stategy.points

    function get_bc_loss_func(bc_loss_func, bc_bound)
        return θ -> begin
            set = NeuralPDE.generate_quasi_random_points(points, bc_bound, eltypeθ,
                                                         sampling_alg)
            set = sortslices(set; dims=2, alg=InsertionSort,
                             lt=(x, y) -> isless(x[tidx], y[tidx]))
            set_ = ChainRulesCore.@ignore_derivatives adapt(device, set)
            abs2.(bc_loss_func(set_, θ))
        end
    end

    bc_loss_functions = [get_bc_loss_func(bc_loss_func, bc_bound)
                         for (bc_loss_func, bc_bound) in zip(datafree_bc_loss_functions,
                                                             bc_bounds)]

    function get_pde_loss_function(datafree_pde_loss_func, pde_bound)
        return θ -> begin
            L_init = reduce(+, [loss_func(θ) for loss_func in init_loss_functions])
            L_bc = reduce((x, y) -> x .+ y,
                          [loss_func(θ) for loss_func in bc_loss_functions])

            set = NeuralPDE.generate_quasi_random_points(points, pde_bound, eltypeθ,
                                                         sampling_alg)
            set = sortslices(set; dims=2, alg=InsertionSort,
                             lt=(x, y) -> isless(x[tidx], y[tidx]))
            set_ = adapt(device, set)
            L_pde = ChainRulesCore.@ignore_derivatives abs2.(datafree_pde_loss_func(set_,
                                                                                    θ))
            L = ChainRulesCore.@ignore_derivatives hcat(adapt(device, [L_init;;]),
                                                        L_bc[:, 1:(end - 1)] .+
                                                        L_pde[:, 1:(end - 1)])
            W = ChainRulesCore.@ignore_derivatives exp.(-ϵ / points .* cumsum(L; dims=2))
            mean(abs2, W .* datafree_pde_loss_func(set_, θ))
        end
    end

    pde_loss_functions = [get_pde_loss_function(pde_loss_func, pde_bound)
                          for (pde_loss_func, pde_bound) in zip(datafree_pde_functions,
                                                                pde_bounds)]
    reduced_bc_loss_functions = [θ -> mean(loss_func(θ)) for loss_func in bc_loss_functions]

    return pde_loss_functions, reduced_bc_loss_functions
end

function get_pde_and_bc_loss_function(init_loss_functions, datafree_pde_functions,
                                      pde_bounds, tidx, eltypeθ, device, ϵ, stategy)
    sampling_alg = stategy.sampling_alg
    points = stategy.points

    function get_pde_loss_function(datafree_pde_loss_func, pde_bound)
        return θ -> begin
            L_init = reduce(+, [loss_func(θ) for loss_func in init_loss_functions])

            set = NeuralPDE.generate_quasi_random_points(points, pde_bound, eltypeθ,
                                                         sampling_alg)
            set = sortslices(set; dims=2, alg=InsertionSort,
                             lt=(x, y) -> isless(x[tidx], y[tidx]))
            set_ = adapt(device, set)
            L_pde = ChainRulesCore.@ignore_derivatives abs2.(datafree_pde_loss_func(set_,
                                                                                    θ))
            L = ChainRulesCore.@ignore_derivatives hcat(adapt(device, [L_init;;]),
                                                        L_pde[:, 1:(end - 1)])
            W = ChainRulesCore.@ignore_derivatives exp.(-ϵ / points .* cumsum(L; dims=2))
            mean(abs2, W .* datafree_pde_loss_func(set_, θ))
        end
    end

    pde_loss_functions = [get_pde_loss_function(pde_loss_func, pde_bound)
                          for (pde_loss_func, pde_bound) in zip(datafree_pde_functions,
                                                                pde_bounds)]

    return pde_loss_functions
end
