"""
    FourierFeature(in_dims::Int; num_modes::Int, std::Number=10.0f0)
    FourierFeature(int_dims::Int, out_dims::Int; std::Number=10.0f0)

Fourier Feature Network.

# Keyword Arguments

- `num_modes`: Number of modes.
- `std`: Standard deviation of the Gaussian distribution from which the frequencies are sampled.

# States

- `modes`: Random Fourier mappings.
"""
struct FourierFeature{S} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    num_modes::Int
    std::S
end

function FourierFeature(in_dims::Int; num_modes::Int, std::Number=10.0f0)
    return FourierFeature(in_dims, num_modes * 2, num_modes, std)
end
function FourierFeature(int_dims::Int, out_dims::Int; std::Number=10.0f0)
    @assert iseven(out_dims) "The output dimension must be even"
    return FourierFeature{typeof(std)}(int_dims, out_dims, out_dims ÷ 2, std)
end

function FourierFeature(ch::Pair{Int, Int}; std::Number=10.0f0)
    return FourierFeature(first(ch), last(ch); std = std)
end

function initialstates(rng::AbstractRNG, f::FourierFeature)
    modes = randn(rng, Float32, f.num_modes, f.in_dims) .* f.std
    return (modes=modes,)
end

function (f::FourierFeature)(x::AbstractVecOrMat, ps, st::NamedTuple)
    x = st.modes * x
    x = 2 * eltype(x)(π) .* x
    return cat(sin.(x), cos.(x); dims=1), st
end

function (f::FourierFeature)(x::AbstractArray, ps, st::NamedTuple)
    x = batched_mul(st.modes, x)
    x = 2 * eltype(x)(π) .* x
    return cat(sin.(x), cos.(x); dims=1), st
end

function Base.show(io::IO, f::FourierFeature)
    return print(io, "FourierFeature($(f.in_dims) => $(f.out_dims))")
end

"""
    TriplewiseFusion(connection, layers...)

```
         u1                    u2
            ↘                     ↘
h1 → layer1 → connection → layer2 → connection
            ↗                     ↗
         v1                    v2
```

## Arguments

  - `connection`: Takes 3 inputs and combines them
  - `layers`: [`AbstractExplicitLayer`](@ref)s

## Inputs

Layer behaves differently based on input type:

 1. A tripe of `(h, u, v)`, where `u` and `v` itself are tuples of length `N`, the `layers` is also a tuple of
    length `N`. The computation is as follows

```julia
for i in 1:N
    h = connection(layers[i](h), u[i], v[i])
end
```

 2. A triple of `(h, u, v)`, where `u` and `v` are `AbstractArray`s.

```julia
for i in 1:N
    h = connection(layers[i](h), u, v)
end
```

## Returns

  - See Inputs section for how the return value is computed
  - Updated model state for all the contained layers

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`
"""
struct TriplewiseFusion{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function TriplewiseFusion(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return TriplewiseFusion(connection, NamedTuple{names}(layers))
end

function (m::TriplewiseFusion)(x::Union{NTuple{3, AbstractArray},
                                        Tuple{AbstractArray, Vararg{Tuple}}}, ps,
                               st::NamedTuple)
    return applytriplewisefusion(m.layers, m.connection, x, ps, st)
end

@generated function applytriplewisefusion(layers::NamedTuple{names}, connection::C, x::T,
                                          ps, st::NamedTuple{names}) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    calls = [:($(y_symbols[N + 1]) = x[1])]
    function getuv(i)
        return T <: Tuple{AbstractArray, Vararg{Tuple}} ? ($x[2][$i], $x[3][$i]) :
               ($x[2], $x[3])
    end
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(y_symbols[N + 1]),
                                                                ps.$(names[i]),
                                                                st.$(names[i]));
               $(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getuv(i)...)))
             for i in 1:N])
    push!(calls, :(st = NamedTuple{$names}(($(Tuple(st_symbols))))))
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::TriplewiseFusion) = Base.keys(getfield(m, :layers))

function FullyConnected end
