struct QuasiRandom{S} <: NeuralPDE.AbstractTrainingStrategy
    pde_points::Vector{Int}
    boundary_points::Vector{Int}
    sampling_alg::S
end

function QuasiRandom(pde_points::Int, boundary_points::Int=pde_points,
                             sampling_alg=LatinHypercubeSample())
    return QuasiRandom{typeof(sampling_alg)}([pde_points], [boundary_points], sampling_alg)
end

function QuasiRandom(pde_points::Vector{Int}, boundary_points::Vector{Int},
                             sampling_alg=LatinHypercubeSample())
    return QuasiRandom{typeof(sampling_alg)}(pde_points, boundary_points, sampling_alg)
end

function QuasiRandom(pde_points::Int, boundary_points::Vector{Int},
                             sampling_alg=LatinHypercubeSample())
    return QuasiRandom{typeof(sampling_alg)}([pde_points], boundary_points, sampling_alg)
end

function QuasiRandom(pde_points::Vector{Int}, boundary_points::Int,
                             sampling_alg=LatinHypercubeSample())
    return QuasiRandom{typeof(sampling_alg)}(pde_points, [boundary_points], sampling_alg)
end

function NeuralPDE.merge_strategy_with_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                                     strategy::QuasiRandom,
                                                     datafree_pde_loss_function,
                                                     datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars, init_params) = pinnrep
    (; pde_points, boundary_points, sampling_alg) = strategy

    pde_points = length(pde_points) == 1 ? fill(pde_points[1], length(eqs)) : pde_points
    boundary_points = length(boundary_points) == 1 ? fill(boundary_points[1], length(bcs)) : boundary_points

    eltypeθ = eltype(pinnrep.init_params)
    device = parameterless_type(getdata(init_params))

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars,
                                  strategy)
    pde_bounds, bcs_bounds = bounds

    pde_datasets = [NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ,
                                                           sampling_alg)
                    for (points, bound) in zip(pde_points, pde_bounds)]
    pde_datasets = [adapt(device, pde_dataset) for pde_dataset in pde_datasets]

    boundary_datasets = [NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ,
                                                                sampling_alg)
                         for (points, bound) in zip(boundary_points, bcs_bounds)]

    boundary_datasets = [adapt(device, boundary_dataset) for boundary_dataset in boundary_datasets]

    pde_loss_functions = [get_loss_function(loss_func, dataset, strategy)
                          for (loss_func, dataset) in zip(datafree_pde_loss_function, pde_datasets)]

    bc_loss_functions = [get_loss_function(loss_func, dataset, strategy)
                         for (loss_func, dataset) in zip(datafree_bc_loss_function, boundary_datasets)]

    return pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, dataset, strategy::QuasiRandom)
    loss(θ) = mean(abs2, loss_function(dataset, θ))
    return loss
end

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy::QuasiRandom)
    dict_span = Dict([Symbol(d.variables) => [
                          infimum(d.domain),
                          supremum(d.domain),
                      ] for d in domains])
    pde_args = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)

    pde_bounds = map(pde_args) do pd
        span = map(p -> get(dict_span, p, p), pd)
        map(s -> adapt(eltypeθ, s), span)
    end

    bound_args = NeuralPDE.get_argument(bcs, dict_indvars, dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])

    bcs_bounds = map(bound_args) do bt
        span = map(b -> get(dict_span, b, b), bt)
        map(s -> adapt(eltypeθ, s), span)
    end
    [pde_bounds, bcs_bounds]
end
