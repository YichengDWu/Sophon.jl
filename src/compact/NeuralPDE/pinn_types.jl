"""
    PINN(chain; device_type::Type=Array{Float64}, kwargs...)
"""
struct PINN{T, PHI, P, K}
    phi::PHI
    init_params::P
    kwargs::K
end

function PINN(chain::NamedTuple; device_type::Type=Array{Float64}, kwargs...)
    phi = ChainState.(chain)
    phi = map(phi) do ϕ
        return ϕ.state = adapt(device, ϕ.state)
    end

    init_params = ComponentArray(initialparameters(Random.default_rng(), phi))
    init_params = adapt(device_type, init_params)

    return PINN{device_type, typeof(phi), typeof(init_params), typeof(kwargs)}(phi,
                                                                               init_params,
                                                                               kwargs)
end

function PINN(chain::AbstractExplicitLayer; device_type::Type=Array{Float64}, kwargs...)
    phi = ChainState(chain)
    phi.state = adapt(device, phi.state)

    init_params = ComponentArray(initialparameters(Random.default_rng(), phi))
    init_params = adapt(device_type, init_params)

    return PINN{device_type, typeof(phi), typeof(init_params), typeof(kwargs)}(phi,
                                                                               init_params,
                                                                               kwargs)
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
function Lux.cpu(c::ChainState)
    return ChainState(c.model, cpu(c.state))
end

function Lux.gpu(c::ChainState)
    return ChainState(c.model, gpu(c.state))
end
