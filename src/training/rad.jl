struct RADTraining <: NeuralPDE.AbstractTrainingStrategy
    points::Int64
    bcs_points::Int64
    k::Float64
    c::Float64
    resample_at::Int64
    training_strategy::QuasiRandomTraining
end

function RADTraining(points; resample_at, k=1.0, c=1.0, sampling_alg=LatinHypercubeSample(),
                     bcs_points=points)
    training_strategy = QuasiRandomTraining(points; bcs_points=bcs_points,
                                            sampling_alg=sampling_alg)
    return RADTraining(points, bcs_points, k, c, resample_at, training_strategy)
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
        pde_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ,
                                                strategy.training_strategy)
                              for (_loss, bound) in zip(datafree_pde_loss_function,
                                                        pde_bounds)]

        strategy_ = QuasiRandomTraining(strategy.training_strategy.bcs_points;
                                        sampling_alg=strategy.training_strategy.sampling_alg)
        bc_loss_functions = [NeuralPDE.get_loss_function(_loss, bound, eltypeθ, strategy_)
                             for (_loss, bound) in zip(datafree_bc_loss_function,
                                                       bcs_bounds)]

        return pde_loss_functions, bc_loss_functions

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
        sets = NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ, sampling_alg)
        sets_ = Adapt.adapt(SciMLBase.parameterless_type(ComponentArrays.getdata(θ)), sets)
        ϵ = (abs.(loss_function(sets_, θ))) .^ k .+ c
        subsets = wsample(set, ϵ, points)
        mean(abs2, loss_function(subsets, θ))
    end
    return loss
end
