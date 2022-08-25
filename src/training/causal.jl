"""
    CausalTraining(points; epsilon, init_points=points, sampling_strategy)

## Keyword arguments

  - `epsilon`: How much you respect causality. If `epsilon` is 0, then it falls back to `sampling_strategy`. You can also pass in a `AbstractSchedule`.
  - `init_points`: Initial points to use. If `init_points` is `nothing`, then it uses `points`.
  - `sampling_strategy`: How to sample points. Recommended to use `GridTraining` as `QuasiRamdonTraining` could be every inefficient.

## References

[1] Wang S, Sankaran S, Perdikaris P. Respecting causality is all you need for training physics-informed neural networks[J]. arXiv preprint arXiv:2203.07404, 2022.
"""
struct CausalTraining{S} <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    init_points::Int64
    epsilon::AbstractSchedule
    sampling_strategy::S
end

function CausalTraining(sampling_strategy; epsilon, init_points=points)
    epsilon = epsilon isa Real ? Constant(Float64(epsilon)) : epsilon
    return CausalTraining{typeof(sampling_strategy)}(points, init_points, epsilon,
                                                     sampling_strategy)
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::CausalTraining{
                                                                              QuasiRandomTraining
                                                                              },
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, dict_indvars, bc_indvars, iteration) = pinnrep

    ϵ = strategy.epsilon(first(iteration)) # Currently not working, there is a bug

    tidx = get(:t, dict_indvars, [])
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
                                    sampling_alg=strategy.sampling_strategy.sampling_alg)
    init_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy_)
                           for (_loss, bound) in zip(datafree_init_loss_function,
                                                     init_bounds)]

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
                                      eltypeθ, device, ϵ,
                                      stategy::CausalTraining{QuasiRandomTraining})
    sampling_alg = stategy.sampling_strategy.sampling_alg
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

function get_pde_loss_function(init_loss_functions, datafree_pde_functions, pde_bounds,
                               tidx, eltypeθ, device, ϵ,
                               stategy::CausalTraining{QuasiRandomTraining})
    sampling_alg = stategy.sampling_strategy.sampling_alg
    points = stategy.points

    function get_loss_function(datafree_pde_loss_func, pde_bound)
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

    pde_loss_functions = [get_loss_function(pde_loss_func, pde_bound)
                          for (pde_loss_func, pde_bound) in zip(datafree_pde_functions,
                                                                pde_bounds)]

    return pde_loss_functions
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::CausalTraining{GridTraining},
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, dict_indvars, bc_indvars, iteration) = pinnrep

    ϵ = strategy.epsilon(first(iteration)) # Currently not working, there is a bug

    tidx = get(:t, dict_indvars, [])
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

    strategy_ = GridTrainingTraining(1 / strategy.init_points) # TODO: change this to T/init_points
    init_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy_)
                           for (_loss, bound) in zip(datafree_init_loss_function,
                                                     init_bounds)]

    pde_train_sets = [generate_pde_training_set(points, pde_bound)
                      for pde_bound in pde_bounds]
    pde_train_sets = [sortslices(set; dims=2, alg=InsertionSort,
                                 lt=(x, y) -> isless(x[tidx], y[tidx]))
                      for set in pde_train_sets]
    pde_train_sets = [adapt(device, set) for set in pde_train_sets]

    if isempty(real_datafree_bc_loss_function) && !isempty(init_loss_functions) # Only this case is implemented
        pde_loss_functions = get_pde_loss_function(init_loss_functions,
                                                   datafree_pde_loss_function,
                                                   pde_train_sets, ϵ, strategy)
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

function get_pde_loss_function(init_loss_functions, datafree_pde_functions, pde_train_sets,
                               ϵ, stategy::CausalTraining{GridTraining})
    points = stategy.points

    function get_loss_function(datafree_pde_loss_func, pde_train_set)
        return θ -> begin
            L_init = reduce(+, [loss_func(θ) for loss_func in init_loss_functions])

            L_pde = ChainRulesCore.@ignore_derivatives abs2.(datafree_pde_loss_func(pde_train_set,
                                                                                    θ))
            L_pde = reshape(L_pde, size(L_pde, 1), :, points) # last dim is time
            s_ = size(L_pde, 2)
            L_pde = mean(L_pde; dims=2) # average loss over sapce points at each time point
            L_pde = cumsum(L_pde; dims=3)
            L = ChainRulesCore.@ignore_derivatives cat(similar(L_pde, 1, 1, 1) .* 0,
                                                       L_pde[:, :, 1:(end - 1)]; dims=3) .+
                                                   L_init

            W = exp.(-ϵ .* L)
            W = repeat(W, 1, s_, 1)
            W = ChainRulesCore.@ignore_derivatives reshape(W, 1, :)
            mean(abs2, W .* datafree_pde_loss_func(pde_train_set, θ))
        end
    end

    pde_loss_functions = [get_loss_function(pde_loss_func, pde_train_set)
                          for (pde_loss_func, pde_train_set) in zip(datafree_pde_functions,
                                                                    pde_train_sets)]

    return pde_loss_functions
end

NeuralPDE.@nograd function generate_pde_training_set(points, pde_bound)
    span = map(b -> range(b[1], b[2], points), pde_bound)
    pde_train_set = hcat(vec(collect.(Iterators.product(span...)))...)
    return pde_train_set
end
