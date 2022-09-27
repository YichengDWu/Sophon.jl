@doc raw"""
    EvoTraining(points; sampling_alg=LatinHypercubeSample(), bcs_points=points, ϵ = 100, η = 1e-3,
    Δ = 0.1)
```math
\beta_{i+1}=\beta_{i}+\eta e^{-\epsilon \mathcal{L}_{r}^{g}(\theta)}
```
## Keyword Arguments

  - `ϵ`:  Tolerance that controls how low the PDE loss needs to be before the gate shifts to the right.
  - `η`: The learning rate of updating `β` that controls how fast the gate should propagate.
  - `Δ`: The maximum allowed magnitude of update of `β`.
  - `λ`: The weights of the initial and boundary conditions.

## References

[1] Daw A, Bu J, Wang S, et al. Rethinking the Importance of Sampling in Physics-informed Neural Networks[J]. arXiv preprint arXiv:2207.02338, 2022.
"""
struct EvoTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    bcs_points::Int64
    sampling_strategy::QuasiRandomTraining
    ϵ::Float64
    η::Float64
    Δ::Float64
    λ::Float64
end

function EvoTraining(points; sampling_alg=LatinHypercubeSample(), bcs_points=points, ϵ=1.0,
                     η=1e-3, Δ=0.1, λ=1.0)
    sampling_strategy = QuasiRandomTraining(points; bcs_points=bcs_points,
                                            sampling_alg=sampling_alg)
    return EvoTraining(points, bcs_points, sampling_strategy, ϵ, η, Δ, λ)
end

mutable struct EvoDataset
    set::Any
    tidx::Int
    T::Float64
    β::Float64
    ϵ::Float64
    η::Float64
    Δ::Float64
end

NeuralPDE.@nograd function generate_and_adapt_set(points, bound, eltypeθ, device,
                                                  sampling_alg)
    set = NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
    set = adapt(device, set)
    return set
end

NeuralPDE.@nograd function updateset!(loss_func, evo::EvoDataset, θ, points, bound, eltypeθ,
                                      device, sampling_alg)
    losses = abs.(loss_func(evo.set, θ))
    t = @view evo.set[[evo.tidx], :]
    gt = Base.Fix2(gate, evo.β).(t)
    fitness = losses .* gt
    set_r = evo.set[:, vec(fitness) .> median(fitness)]
    set_s = generate_and_adapt_set(points - size(set_r, 2), bound, eltypeθ, device,
                                   sampling_alg)
    set_total = hcat(set_r, set_s)
    evo.set = set_total

    evo.β = evo.β + evo.η * min(exp(-evo.ϵ * mean(fitness)), evo.Δ)
    new_t = @view set_total[[evo.tidx], :]
    new_gt = Base.Fix2(gate, evo.β).(new_t)
    return set_total, new_gt
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::EvoTraining,
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params, bc_indvars) = pinnrep

    tidx = dict_indvars[:t]

    eltypeθ = eltype(ComponentArrays.getdata(pinnrep.flat_init_params))
    device = SciMLBase.parameterless_type(ComponentArrays.getdata(flat_init_params))
    (; ϵ, η, Δ, λ, sampling_strategy) = strategy
    sampling_alg = sampling_strategy.sampling_alg

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    pde_datasets = [EvoDataset(generate_and_adapt_set(strategy.points, bound, eltypeθ,
                                                      device, sampling_alg), tidx, 1.0,
                               -0.5, ϵ, η, Δ) for bound in pde_bounds]

    pde_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy, set)
                          for (_loss, bound, set) in zip(datafree_pde_loss_function,
                                                         pde_bounds, pde_datasets)]

    bc_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ,
                                                     sampling_strategy)
                         for (_loss, bound) in zip(datafree_bc_loss_function, bcs_bounds)]

    pde_loss_functions, bc_loss_functions
    return [θ -> λ * loss_func(θ) for loss_func in bc_loss_functions], pde_loss_functions
end

function get_loss_function(loss_function, bound, eltypeθ, strategy::EvoTraining,
                           set::EvoDataset)
    sampling_alg = strategy.sampling_strategy.sampling_alg
    points = strategy.points

    loss = θ -> begin
        device = SciMLBase.parameterless_type(ComponentArrays.getdata(θ))
        new_set, gt = updateset!(loss_function, set, θ, points, bound, eltypeθ, device,
                                 sampling_alg)
        mean(abs2.(loss_function(new_set, θ)) .* gt)
    end
    return loss
end

gate(t, β=-0.5) = (1 - NNlib.tanh_fast(5 * (t - β))) / 2
