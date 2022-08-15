"""
    CausalTraining(points; epsilon, bcs_points=points, sampling_alg=LatinHypercubeSample())

## Keyword arguments

- `epsilon`: How much you respect causality.

## References

[1] Wang S, Sankaran S, Perdikaris P. Respecting causality is all you need for training physics-informed neural networks[J]. arXiv preprint arXiv:2203.07404, 2022.
"""
struct CausalTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    bcs_points::Int64
    epsilon::Float64
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
end

function CausalTraining(points; epsilon, bcs_points=points, sampling_alg=LatinHypercubeSample())
    return CausalTraining(points, bcs_points, epsilon, sampling_alg)
end

ChainRulesCore.@ignore_derivatives function generate_quasi_random_points(points, bound,
                                                                         eltypeθ,
                                                                         sampling_alg)
    function f(b)
        if b isa Number
            fill(eltypeθ(b), (1, points))
        else
            lb, ub = eltypeθ[b[1]], [b[2]]
            QuasiMonteCarlo.sample(points, lb, ub, sampling_alg)
        end
    end
    return vcat(f.(bound)...)
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::CausalTraining,
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (;domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, dict_indvars) = pinnrep

    tidx = dict_indvars[:t]
    eltypeθ = eltype(flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    pde_loss_functions = [get_temporal_loss_function(_loss, bound, tidx, eltypeθ, strategy)
                          for (_loss, bound) in zip(datafree_pde_loss_function, pde_bounds)]

    strategy_ = QuasiRandomTraining(strategy.bcs_points; sampling_alg=strategy.sampling_alg)
    bc_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy_)
                         for (_loss, bound) in zip(datafree_bc_loss_function, bcs_bounds)]

    return pde_loss_functions, bc_loss_functions
end

function get_temporal_loss_function(loss_function, bound, tidx, eltypeθ, strategy::CausalTraining)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    ϵ = strategy.epsilon

    loss = θ -> begin
            set = NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
            W, set_ = get_causal_weights(loss_function, θ, SciMLBase.parameterless_type(ComponentArrays.getdata(θ)), set, tidx, ϵ)
            mean(abs2, W .* loss_function(set_, θ))
        end
    return loss
end

NeuralPDE.@nograd function get_causal_weights(loss_function, θ, type_, set, tidx, ϵ)
    set = sortslices(set, dims=2, alg=InsertionSort, lt=(x,y)->isless(x[tidx],y[tidx]))
    set_ = adapt(type_, set)
    L = abs2.(loss_function(set_, θ))
    W = exp.(- ϵ .* cumsum(L, dims = 2))
    W = hcat(adapt(type_,[one(eltype(W));;]), W[:,1:end-1])
    return W, set_
end
