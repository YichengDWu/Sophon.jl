struct FourierFeature <: AbstractExplicitLayer
    in_dims::Any
    out_dims::Any
    num_modes::Any
    std::Any
end

function FourierFeature(in_dims::Int; num_modes::Int, std::Number=10.0f0)
    return FourierFeature(in_dims, num_modes * 2, num_modes, std)
end
function FourierFeature(int_dims::Int, out_dims::Int; std::Number=10.0f0)
    @assert iseven(out_dims) "The output dimension must be even"
    return FourierFeature(int_dims, out_dims, out_dims ÷ 2, std)
end

function FourierFeature(ch::Pair{Int, Int}; std::Number=10.0f0)
    return FourierFeature(first(ch), last(ch); std)
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
