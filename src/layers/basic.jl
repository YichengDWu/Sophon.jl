@doc raw"""
    FourierFeature(in_dims::Int, modes::NTuple{N,Pair{S,T}}) where {N,S,T<:Int}

Fourier Feature Network.

```math
\phi^{(i)}(x)=\left[\sin \left(2 \pi W^{(i)} x\right) ; \cos 2 \pi W^{(i)} x\right], \quad W^{(i)} \sim \mathcal{N}\left(0, \sigma^{(i)}\right)
```

# Arguments

  - `in_dims`: Number of the input dimensions.
  - `modes`: A tuple of pairs of `std => out_dims`, where `std` is the standard deviation of
    the Gaussian distribution, and `out_dims` is the corresponding number of output dimensions.

# Inputs

  - `x`: An an AbstractArray with `size(x, 1) == in_dims`.

# Returns

  - An `AbstractArray` with `size(y, 1) == sum(last(modes) * 2)`.
  - The states of the modes.

# States

  - States of each mode wrapped in a NamedTuple with `fields = mode_1, mode_2, ..., mode_N`.

# Examples

```julia
julia> f = FourierFeature(2, (1 => 3, 50 => 4))
FourierFeature(2 => 14)

julia> ps, st = Lux.setup(rng, f)
(NamedTuple(), (mode_1 = Float32[0.7510394 0.0678698; -1.6466209 -0.08511321; -0.4704813 2.0663197], mode_2 = Float32[-98.90031 -42.593884; 110.35572 15.565719; 81.60114 51.257904; -0.53021294 15.216658]))

julia> st
(mode_1 = Float32[0.7510394 0.0678698; -1.6466209 -0.08511321; -0.4704813 2.0663197], mode_2 = Float32[-98.90031 -42.593884; 110.35572 15.565719; 81.60114 51.257904; -0.53021294 15.216658])
```

# References

[1] Tancik, Matthew, et al. “Fourier features let networks learn high frequency functions in low dimensional domains.” Advances in Neural Information Processing Systems 33 (2020): 7537-7547.
"""
struct FourierFeature{M} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    modes::M
end

function Base.show(io::IO, f::FourierFeature)
    return print(io, "FourierFeature($(f.in_dims) => $(f.out_dims))")
end

function FourierFeature(in_dims::Int, modes::NTuple{N, Pair{S, T}}) where {N, S, T <: Int}
    out_dims = map(x -> 2 * last(x), modes)
    return FourierFeature{typeof(modes)}(in_dims, sum(out_dims), modes)
end

function FourierFeature(in_dims::Int, modes::NTuple{N, T}) where {N, T <: Real}
    out_dims = length(modes) * 2 * in_dims
    return FourierFeature{typeof(modes)}(in_dims, out_dims, modes)
end

function initialstates(rng::AbstractRNG,
                       f::FourierFeature{NTuple{NN, Pair{S, T}}}) where {NN, S, T <: Int}
    N = length(f.modes)
    names = ntuple(i -> Symbol("mode_$i"), N)
    frequencies = ntuple(N) do i
        m = f.modes[i]
        return randn(rng, Float32, last(m), f.in_dims) .* first(m)
    end
    return NamedTuple{names}(frequencies)
end

function initialstates(rng::AbstractRNG,
                       f::FourierFeature{NTuple{NN, T}}) where {NN, T <: Real}
    N = length(f.modes)
    names = ntuple(i -> Symbol("mode_$i"), N)
    return NamedTuple{names}(f.modes)
end

function (f::FourierFeature)(x::AbstractVecOrMat, ps, st::NamedTuple)
    x = 2 * eltype(x)(π) .* x
    y = mapreduce(vcat, st) do f
        return [sin.(f * x); cos.(f * x)]
    end
    return y, st
end

function (f::FourierFeature)(x::AbstractArray, ps, st::NamedTuple)
    x = 2 * eltype(x)(π) .* x
    y = mapreduce(vcat, st) do f
        return [sin.(batched_mul(f, x)); cos.(batched_mul(f, x))]
    end
    return y, st
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

  - `connection`: A functio takes 3 inputs and combines them.
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
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::TriplewiseFusion) = Base.keys(getfield(m, :layers))

"""
    Sine(in_dims::Int, out_dims::Int; omega::Real)

Sinusoidal layer.

## Example

```julia
s = Sine(2, 2; omega=30.0f0) # first layer
s = Sine(2, 2) # hidden layer
```
"""
function Sine(ch::Pair{T, T}; omega::Union{Real, Nothing}=nothing) where {T <: Int}
    return Sine(first(ch), last(ch); omega=omega)
end

function Sine(in_dims::Int, out_dims::Int; omega::Union{Real, Nothing}=nothing)
    init_weight = get_sine_init_weight(omega)
    return Dense(in_dims, out_dims, sin; init_weight=init_weight)
end

get_sine_init_weight(::Nothing) = kaiming_uniform(sin)
function get_sine_init_weight(omega::Real)
    function init_weight(rng::AbstractRNG, out_dims, in_dims)
        weight = (rand(rng, Float32, out_dims, in_dims) .- 0.5f0) .* 2.0f0
        scale = Float32(omega) / in_dims
        return weight .* scale
    end
    return init_weight
end
"""
    RBF(in_dims::Int, out_dims::Int, num_centers::Int=out_dims; sigma::AbstractFloat=0.2f0)

Radial Basis Fuction Network.
"""
struct RBF{F1, F2} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    num_centers::Int
    sigma::Float32
    init_center::F1
    init_weight::F2
end

function RBF(in_dims::Int, out_dims::Int, num_centers::Int=out_dims,
             sigma::AbstractFloat=0.2f0, init_center=Lux.glorot_uniform,
             init_weight=Lux.glorot_normal)
    return RBF{typeof(init_center), typeof(init_weight)}(in_dims, out_dims, num_centers,
                                                         sigma, init_center, init_weight)
end

function RBF(mapping::Pair{<:Int, <:Int}, num_centers::Int=out_dims,
             sigma::AbstractFloat=0.2f0, init_center=Lux.glorot_uniform,
             init_weight=Lux.glorot_uniform)
    return RBF(first(mapping), last(mapping), num_centers, sigma, init_center, init_weight)
end

function initialparameters(rng::AbstractRNG, s::RBF)
    center = s.init_center(rng, s.num_centers, s.in_dims)
    weight = s.init_weight(rng, s.out_dims, s.num_centers)
    return (center=center, weight=weight)
end

function (rbf::RBF)(x::AbstractVecOrMat, ps, st::NamedTuple)
    x_norm = sum(abs2, x; dims=1)
    center_norm = sum(abs2, ps.center; dims=2)
    d = -2 * ps.center * x .+ x_norm .+ center_norm
    z = -1 / rbf.sigma .* d
    z_shit = z .- maximum(z; dims=1)
    r = exp.(z_shit)
    r = r ./ reshape(sum(r; dims=1), 1, :)
    y = ps.weight * r
    return y, st
end

function Base.show(io::IO, rbf::RBF)
    return print(io, "RBF($(rbf.in_dims) => $(rbf.out_dims))")
end
