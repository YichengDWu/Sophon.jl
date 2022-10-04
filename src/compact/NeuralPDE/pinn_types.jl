"""
    PINN(chain, rng::AbstractRNG=Random.default_rng())

A container for a neural network, its states and its initial parameters. Call `gpu` and `cpu` to move the neural network to the GPU and CPU respectively.
The default element type of the neural network is `Float32`.

## Arguments

  - `chain`: `AbstractExplicitLayer` or a named tuple of `AbstractExplicitLayer`s.
  - `rng`: `AbstractRNG` to use for initialising the neural network.
"""
struct PINN{PHI, P}
    phi::PHI
    init_params::P
end

function PINN(rng::AbstractRNG=Random.default_rng(); kwargs...)
    return PINN((; kwargs...), rng)
end

function PINN(chain::NamedTuple, rng::AbstractRNG=Random.default_rng())
    phi = map(m -> ChainState(m, rng), chain)
    init_params = ComponentArray(initialparameters(rng, phi)) .|> Float64

    return PINN{typeof(phi), typeof(init_params)}(phi, init_params)
end

function PINN(chain::AbstractExplicitLayer, rng::AbstractRNG=Random.default_rng())
    phi = ChainState(chain, rng)
    init_params = ComponentArray(initialparameters(rng, phi)) .|> Float64

    return PINN{typeof(phi), typeof(init_params)}(phi, init_params)
end

"""
    ChainState(model, rng::AbstractRNG=Random.default_rng())

Wraps a model in a stateful container.

## Arguments

    - `model`: `AbstractExplicitLayer`, or a named tuple of them, which will be treated as a `Chain`.
"""
mutable struct ChainState{L, S}
    model::L
    state::S
end

function ChainState(model, rng::AbstractRNG=Random.default_rng())
    states = initialstates(rng, model)
    return ChainState{typeof(model), typeof(states)}(model, states)
end

function ChainState(model, state::NamedTuple)
    return ChainState{typeof(model), typeof(state)}(model, state)
end

function ChainState(; rng::AbstractRNG=Random.default_rng(), kwargs...)
    return ChainState((; kwargs...), rng)
end

@inline ChainState(a::ChainState) = a

@inline function initialparameters(rng::AbstractRNG, s::ChainState)
    return initialparameters(rng, s.model)
end

function (c::ChainState{<:NamedTuple})(x, ps)
    y, st = Lux.applychain(c.model, x, ps, c.state)
    ChainRulesCore.@ignore_derivatives c.state = st
    return y
end

function (c::ChainState{<:AbstractExplicitLayer})(x, ps)
    y, st = c.model(x, ps, c.state)
    ChainRulesCore.@ignore_derivatives c.state = st
    return y
end

const NTofChainState{names} = NamedTuple{names, <:Tuple{Vararg{ChainState}}}

# construct a new ChainState
function Lux.cpu(cs::ChainState)
    Lux.@set! cs.state = cpu(cs.state)
    return cs
end

function Lux.gpu(cs::ChainState)
    Lux.@set! cs.state = adapt(CuArray, cs.state)
    return cs
end

function Lux.cpu(cs::NamedTuple{names, <:Tuple{Vararg{ChainState}}}) where {names}
    return map(cs) do c
        return cpu(c)
    end
end

function Lux.gpu(cs::NamedTuple{names, <:Tuple{Vararg{ChainState}}}) where {names}
    return map(cs) do c
        return gpu(c)
    end
end

function Lux.gpu(pinn::PINN)
    Lux.@set! pinn.phi = gpu(pinn.phi)
    Lux.@set! pinn.init_params = adapt(CuArray, pinn.init_params)
    return pinn
end

function Lux.cpu(pinn::PINN)
    Lux.@set! pinn.phi = cpu(pinn.phi)
    Lux.@set! pinn.init_params = cpu(pinn.init_params)
    return pinn
end
