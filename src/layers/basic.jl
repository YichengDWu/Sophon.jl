"""
    FourierFeature(in_dims::Int, modes::NTuple{N,Pair{S,T}}) where {N,S,T<:Int}

Fourier Feature Network.

# Arguments

  - `int_dims`: Input dimension.
  - `modes`: A tuple of pairs of `std => out_dims`, where `std` is the standard deviation of the Gaussian distribution, and `out_dims` is the output dimension.

# Inputs

  - `x`: An an AbstractArray with `size(x, 1) == in_dims`.

# Returns

  - An AbstractArray with `size(y, 1) == sum(last(modes) * 2)`.
  - The states of the layers.

# States

  - `modes`: Random Fourier mappings.

# Examples

```julia
FourierFeature(2, (1 => 3, 50 => 4))
```

# References

[1] Tancik, Matthew, et al. “Fourier features let networks learn high frequency functions in low dimensional domains.” Advances in Neural Information Processing Systems 33 (2020): 7537-7547.
"""
struct FourierFeature{M} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    modes::M
end

function FourierFeature(in_dims::Int, modes::NTuple{N, Pair{S, T}}) where {N, S, T <: Int}
    out_dims = map(x -> 2 * last(x), modes)
    return FourierFeature{typeof(modes)}(in_dims, sum(out_dims), modes)
end

function initialstates(rng::AbstractRNG, f::FourierFeature)
    N = length(f.modes)
    names = ntuple(i -> Symbol("mode_$i"), N)
    frequencies = ntuple(N) do i
        m = f.modes[i]
        return randn(rng, Float32, last(m), f.in_dims) .* first(m)
    end
    return NamedTuple{names}(frequencies)
end

function (f::FourierFeature)(x::AbstractVecOrMat, ps, st::NamedTuple)
    frequencies = reduce(vcat, st)
    x = 2 * eltype(x)(π) .* frequencies * x
    return vcat(sin.(x), cos.(x)), st
end

function (f::FourierFeature)(x::AbstractArray, ps, st::NamedTuple)
    frequencies = reduce(vcat, st)
    x = 2 * eltype(x)(π) .* batched_mul(frequencies, x)
    return vcat(sin.(x), cos.(x)), st
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
        return T <: Tuple{AbstractArray, Vararg{Tuple}} ? (:(x[2][$i]), :(x[3][$i])) :
               (:(x[2]), :(x[3]))
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

"""
    FullyConnected(int_dims::Int, out_dims::NTuple{N, Int}, activation)

Create fully connected layers.
"""
@generated function FullyConnected(int_dims::Int, out_dims::NTuple{N, T},
                                   activation::Function=identity) where {N, T <: Int}
    N == 1 && return :(Dense(int_dims, out_dims[1], activation))
    get_layer(i) = :(Dense(out_dims[$i] => out_dims[$(i + 1)], activation))
    layers = [:(Dense(int_dims => out_dims[1], activation))]
    append!(layers, [get_layer(i) for i in 1:(N - 2)])
    append!(layers, [:(Dense(out_dims[N - 1] => out_dims[N]))])
    return :(Chain($(layers...)))
end
