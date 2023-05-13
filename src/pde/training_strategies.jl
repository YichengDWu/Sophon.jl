abstract type AbstractTrainingAlg end

"""
    NonAdaptiveTraining(pde_weights=1, bcs_weights=pde_weights)

Fixed weights for the loss functions.

## Arguments

  - `pde_weights`: weights for the PDE loss functions. If a single number is given, it is used for all PDE loss functions.
  - `bcs_weights`: weights for the boundary conditions loss functions. If a single number is given, it is used for all boundary conditions loss functions.
"""
struct NonAdaptiveTraining{P, B} <: AbstractTrainingAlg
    pde_weights::P
    bcs_weights::B
    function NonAdaptiveTraining(pde_weights=1, bcs_weights=pde_weights)
        return new{typeof(pde_weights), typeof(bcs_weights)}(pde_weights, bcs_weights)
    end
end

function scalarize(strategy::NonAdaptiveTraining{P, B}, phi, datafree_pde_loss_functions,
                   datafree_bc_loss_functions) where {P, B}
    (; pde_weights, bcs_weights) = strategy

    N1 = length(datafree_pde_loss_functions)
    N2 = length(datafree_bc_loss_functions)

    pde_weights = P <: Number ? ntuple(_ -> first(pde_weights), N1) : Tuple(pde_weights)
    bcs_weights = B <: Number ? ntuple(_ -> first(bcs_weights), N2) : Tuple(bcs_weights)

    f = scalarize((pde_weights..., bcs_weights...),
                  (datafree_pde_loss_functions..., datafree_bc_loss_functions...))

    return f
end

function scalarize(weights::NTuple{N, <:Real}, datafree_loss_functions::Tuple) where {N}
    body = Expr(:block)
    push!(body.args, Expr(:(=), :local_ps, :(get_local_ps(pp))))
    push!(body.args, Expr(:(=), :global_ps, :(get_global_ps(pp))))

    ex = :(mean($(weights[1]) .*
                abs2.($(datafree_loss_functions[1])(local_ps[1], θ, global_ps)))) # ugly hack to be compatible with PI-Neural Operators
    for i in 2:N                                                                  # local_ps is just coord
        ex = :(mean($(weights[i]) .*
                    abs2.($(datafree_loss_functions[i])(local_ps[$i], θ, global_ps))) + $ex)
    end
    push!(body.args, ex)
    loss_f = Expr(:function, Expr(:call, :(pinn_loss_function), :θ, :pp), body)
    return eval(loss_f)
end

"""
    AdaptiveTraining(pde_weights, bcs_weights)

Adaptive weights for the loss functions. Here `pde_weights` and `bcs_weights` are
functions that take in `(phi, x, θ)` and return the point-wise weights. Note that `bcs_weights` can be
real numbers but they will be converted to functions that return the same numbers.
"""
struct AdaptiveTraining{P, B} <: AbstractTrainingAlg
    pde_weights::P
    bcs_weights::B
end

function AdaptiveTraining(pde_weights::Function, bcs_weights::Real)
    _bcs_weights = (phi, coord, θ) -> bcs_weights
    return AdaptiveTraining{typeof(pde_weights), typeof(_bcs_weights)}(pde_weights,
                                                                       _bcs_weights)
end

function AdaptiveTraining(pde_weights::Function, bcs_weights::NTuple{N, <:Real}) where {N}
    _bcs_weights = map(w -> (phi, coord, θ) -> w, bcs_weights)
    return AdaptiveTraining{typeof(pde_weights), typeof(_bcs_weights)}(pde_weights,
                                                                       _bcs_weights)
end

function AdaptiveTraining(pde_weights::Tuple{Vararg{Function}}, bcs_weights::Int)
    _bcs_weights = (phi, coord, θ) -> bcs_weights
    return AdaptiveTraining{typeof(pde_weights), typeof(_bcs_weights)}(pde_weights,
                                                                       _bcs_weights)
end

function AdaptiveTraining(pde_weights::Tuple{Vararg{Function}},
                          bcs_weights::NTuple{N, <:Real}) where {N}
    _bcs_weights = map(w -> (phi, coord, θ) -> w, bcs_weights)
    return AdaptiveTraining{typeof(pde_weights), typeof(_bcs_weights)}(pde_weights,
                                                                       _bcs_weights)
end

function scalarize(strategy::AdaptiveTraining, phi, datafree_pde_loss_function,
                   datafree_bc_loss_function)
    (; pde_weights, bcs_weights) = strategy

    N1 = length(datafree_pde_loss_function)
    N2 = length(datafree_bc_loss_function)

    pde_weights = pde_weights isa Function ? ntuple(_ -> pde_weights, N1) : pde_weights
    bcs_weights = bcs_weights isa Function ? ntuple(_ -> bcs_weights, N2) : bcs_weights

    f = scalarize(phi, (pde_weights..., bcs_weights...),
                  (datafree_pde_loss_function..., datafree_bc_loss_function...))

    return f
end

function scalarize(phi, weights::Tuple{Vararg{Function}}, datafree_loss_function::Tuple)
    N = length(datafree_loss_function)
    body = Expr(:block)
    push!(body.args, Expr(:(=), :local_ps, :(get_local_ps(pp))))
    push!(body.args, Expr(:(=), :global_ps, :(get_global_ps(pp))))

    ex = :(mean($(weights[1])($phi, local_ps[1], θ) .*
                abs2.($(datafree_loss_function[1])(local_ps[1], θ, global_ps))))
    for i in 2:N
        ex = :(mean($(weights[i])($phi, local_ps[$i], θ) .*
                    abs2.($(datafree_loss_function[i])(local_ps[$i], θ, global_ps))) + $ex)
    end
    push!(body.args, ex)
    loss_f = Expr(:function, Expr(:call, :(pinn_loss_function), :θ, :pp), body)
    return eval(loss_f)
end
