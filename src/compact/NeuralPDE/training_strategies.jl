abstract type AbstractTrainingAlg end

"""
    NonAdaptiveTraining(pde_weights=1, bcs_weights=pde_weights)

Fixed weights for the loss functions.
"""
struct NonAdaptiveTraining{P, B} <: AbstractTrainingAlg
    pde_weights::P
    bcs_weights::B
    function NonAdaptiveTraining(pde_weights=1, bcs_weights=pde_weights)
        return new{typeof(pde_weights), typeof(bcs_weights)}(pde_weights, bcs_weights)
    end
end

function get_pde_and_bcs_loss_function(strategy::NonAdaptiveTraining{P, B},
                                       datafree_pde_loss_function,
                                       datafree_bc_loss_function) where {P, B}
    (; pde_weights, bcs_weights) = strategy

    N1 = length(datafree_pde_loss_function)
    N2 = length(datafree_bc_loss_function)

    pde_weights = P <: Number ? ntuple(_ -> first(pde_weights), N1) : Tuple(pde_weights)
    bcs_weights = B <: Number ? ntuple(_ -> first(bcs_weights), N2) : Tuple(bcs_weights)

    f = get_pde_and_bcs_loss_function((pde_weights..., bcs_weights...),
                                      (datafree_pde_loss_function...,
                                       datafree_bc_loss_function...))

    return f
end

function get_pde_and_bcs_loss_function(weights::NTuple{N},
                                       datafree_loss_function::Tuple) where {N}
    ex = :(mean($(weights[1]) .* abs2.($(datafree_loss_function[1])(p[1], θ))))
    for i in 2:N
        ex = :(mean($(weights[i]) .* abs2.($(datafree_loss_function[i])(p[$i], θ))) + $ex)
    end
    loss_f = :((θ, p) -> $ex)
    return NeuralPDE.@RuntimeGeneratedFunction(loss_f)
end
