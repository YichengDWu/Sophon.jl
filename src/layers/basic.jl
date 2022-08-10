"""
    FourierFeature(in_dims::Int, modes::NTuple{N,Pair{S,T}}) where {N,S,T<:Int}

Fourier Feature Network.

# Arguments

  - `in_dims`: Input dimension.
  - `modes`: A tuple of pairs of `std => out_dims`, where `std` is the standard deviation of the Gaussian distribution, and `out_dims` is the output dimension.

# Inputs

  - `x`: An an AbstractArray with `size(x, 1) == in_dims`.

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
  - `layers`: `AbstractExplicitLayer`s or a `Chain`.

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

function TriplewiseFusion(connection, layers::AbstractExplicitLayer...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return TriplewiseFusion(connection, NamedTuple{names}(layers))
end

function TriplewiseFusion(connection, chain::Chain)
    layers = chain.layers
    return TriplewiseFusion{typeof(connection), typeof(layers)}(connection, layers)
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
    FullyConnected(in_dims, hidden_dims::NTuple{N, Int}, activation; outermost = false)
    FullyConnected(in_dims, hidden_dims, num_layers, activation; outermost = false)

Create fully connected layers.

## Arguments

  - `in_dims`: Input dimension.
  - `hidden_dims`: Hidden dimensions.
  - `num_layers`: Number of layers.
  - `activation`: Activation function.

## Keyword Arguments

  - `outermost`: Whether to use activation function for the last layer.
"""
function FullyConnected(in_dims::Int, hidden_dims::NTuple{N, T}, activation::Function;
                        outermost::Bool=false) where {N, T <: Int}
    return FullyConnected(in_dims, hidden_dims, activation, Val(outermost))
end

function FullyConnected(in_dims::Int, hidden_dims::Int, num_layers::Int, activation;
                        outermost=false)
    return FullyConnected(in_dims, ntuple(i -> hidden_dims, num_layers), activation,
                          Val(outermost))
end

@generated function FullyConnected(in_dims::Int, hidden_dims::NTuple{N, T},
                                   activation::Function, ::Val{F}) where {N, T <: Int, F}
    N == 1 && return :(Dense(in_dims, hidden_dims[1], activation))
    get_layer(i) = :(Dense(hidden_dims[$i] => hidden_dims[$(i + 1)], activation))
    layers = [:(Dense(in_dims => hidden_dims[1], activation))]
    append!(layers, [get_layer(i) for i in 1:(N - 2)])
    append!(layers,
            F ? [get_layer(N - 1)] : [:(Dense(hidden_dims[$(N - 1)] => hidden_dims[$N]))])
    return :(Chain($(layers...)))
end

"""
    Sine(in_dims::Int, out_dims::Int, activation=sin; is_first::Bool = false, omega::AbstractFloat = 30f0)

Sinusoidal layer.

## Example

```julia
s = Sine(2, 2; is_first=true) # first layer
s = Sine(2, 2) # hidden layer
s = Sine(2, 2, identity) # last layer
```
"""
struct Sine{is_first, F} <: AbstractExplicitLayer
    activation::F
    in_dims::Int
    out_dims::Int
    init_omega::Union{Function,Nothing}
end

function Base.show(io::IO, s::Sine)
    return print(io, "Sine($(s.in_dims) => $(s.out_dims))")
end

function Sine(in_dims::Int, out_dims::Int, activation=sin; is_first::Bool=false,
              omega=30.0f0)
    init_omega = is_first ? () -> omega : nothing
    return Sine{is_first, typeof(activation)}(activation, in_dims, out_dims, init_omega)
end

function Sine(chs::Pair{T, T}, activation=sin; is_first::Bool=false,
              omega::AbstractFloat=30.0f0) where {T <: Int}
    return Sine(first(chs), last(chs), activation; is_first=is_first, omega=omega)
end

function initialparameters(rng::AbstractRNG, s::Sine{is_first}) where {is_first}
    weight = (rand(rng, Float32, s.out_dims, s.in_dims) .- 0.5f0) .* 2f0
    scale = is_first ? Float32(s.init_omega()) / s.in_dims : sqrt(6f0 / s.in_dims)
    bias = Lux.zeros32(rng, s.out_dims, 1)
    return (weight=weight .* scale, bias=bias)
end

function initialstates(rng::AbstractRNG, s::Sine{true})
    return (omega=s.init_omega(),)
end

function initialstates(rng::AbstractRNG, s::Sine{false})
    return NamedTuple()
end

function parameterlength(s::Sine)
    return s.out_dims * s.in_dims
end

statelength(s::Sine{true}) = 1
statelength(s::Sine{false}) = 0

@inline function (m::Sine)(x::AbstractVector, ps, st::NamedTuple)
    return m.activation.(ps.weight * x .+ vec(ps.bias)), st
end

@inline function (m::Sine)(x::AbstractMatrix, ps, st::NamedTuple)
    return m.activation.(ps.weight * x .+ ps.bias), st
end

@inline function (m::Sine)(x::AbstractArray, ps, st::NamedTuple)
    return m.activation.(batched_m(ps.weight, x) .+ ps.bias), st
end

@inline function (m::Sine{false, typeof(identity)})(x::AbstractVector, ps, st::NamedTuple)
    return ps.weight * x .+ vec(ps.bias), st
end

@inline function (m::Sine{false, typeof(identity)})(x::AbstractMatrix, ps, st::NamedTuple)
    return ps.weight * x .+ ps.bias, st
end

@inline function (m::Sine{false, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    return batched_m(ps.weight, x) .+ ps.bias, st
end
