"""
    ChainState(model, rng::AbstractRNG=Random.default_rng())
## Arguments
    - `model`: `AbstractExplicitLayer`, or a named tuple of them.
"""
struct ChainState{L, S}
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

@inline function initialparameters(rng::AbstractRNG, s::ChainState)
    return initialparameters(rng, s.model)
end

function (c::ChainState{<:NamedTuple})(x, ps)
    y, st = Lux.applychain(c.model, x, ps, c.state)
    ChainRulesCore.@ignore_derivatives @set! c.state = st
    return y
end

function (c::ChainState{<:AbstractExplicitLayer})(x, ps)
    y, st = c.model(x, ps, c.state)
    ChainRulesCore.@ignore_derivatives @set! c.state = st
    return y
end

function (c::ChainState{<:NamedTuple})(f::Symbol, x, ps)
    y, st = Lux.applychain(c.model.f, x, ps, c.state.f)
    ChainRulesCore.@ignore_derivatives @set! c.state.f = st
    return y
end
