"""
    FourierFeature(in_dim::Int, modes::NTuple{N,Pair{S,T}}) where {N,S,T<:Int}

Fourier Feature Network.

# Arguments

  - `in_dim`: Input dimension.
  - `modes`: A tuple of pairs of `std => out_dim`, where `std` is the standard deviation of the Gaussian distribution, and `out_dim` is the output dimension.

# Inputs

  - `x`: An an AbstractArray with `size(x, 1) == in_dim`.

# Returns

  - An AbstractArray with `size(y, 1) == sum(last(modes) * 2)`.
  - The states of the layers.

# States

  - `modes`: Random frequencies.

# Examples

```julia
FourierFeature(2, (1 => 3, 50 => 4))
```

# References

[1] Tancik, Matthew, et al. “Fourier features let networks learn high frequency functions in low dimensional domains.” Advances in Neural Information Processing Systems 33 (2020): 7537-7547.
"""
struct FourierFeature{M} <: AbstractExplicitLayer
    in_dim::Int
    out_dim::Int
    modes::M
end

function FourierFeature(in_dim::Int, modes::NTuple{N, Pair{S, T}}) where {N, S, T <: Int}
    out_dims = map(x -> 2 * last(x), modes)
    return FourierFeature{typeof(modes)}(in_dim, sum(out_dims), modes)
end

function initialstates(rng::AbstractRNG, f::FourierFeature)
    N = length(f.modes)
    names = ntuple(i -> Symbol("mode_$i"), N)
    frequencies = ntuple(N) do i
        m = f.modes[i]
        return randn(rng, Float32, last(m), f.in_dim) .* first(m)
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
    return print(io, "FourierFeature($(f.in_dim) => $(f.out_dim))")
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
    FullyConnected(in_dim, hidden_dims::NTuple{N, Int}, activation = identity)
    FullyConnected(in_dim, hidden_dims, num_layers, activation=identity)

Create fully connected layers. Note that the last layer is activated as well.
"""
@generated function FullyConnected(in_dim::Int, hidden_dims::NTuple{N, T},
                                   activation::Function=identity) where {N, T <: Int}
    N == 1 && return :(Dense(in_dim, hidden_dims[1], activation))
    get_layer(i) = :(Dense(hidden_dims[$i] => hidden_dims[$(i + 1)], activation))
    layers = [:(Dense(in_dim => hidden_dims[1], activation))]
    append!(layers, [get_layer(i) for i in 1:(N - 1)])
    return :(Chain($(layers...)))
end

function FullyConnected(in_dim::Int, hidden_dim::Int, num_layers::Int, activation=identity)
    return FullyConnected(in_dim, ntuple(i -> hidden_dim, num_layers), activation)
end
