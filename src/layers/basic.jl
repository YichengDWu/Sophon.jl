struct Fourier <: AbstractExplicitLayer
    in_dims
    out_dims
    num_modes
    std
end

Fourier(in_dims::Int; num_modes::Int, std::Number = 1) = Fourier(in_dims, num_modes * 2, num_modes, std)
function Fourier(int_dims::Int, out_dims::Int; std::Number = 1)
    @assert iseven(out_dims) "The output dimension must be even"
    Fourier(int_dims, out_dims, out_dims ÷ 2, std)
end

function Fourier(ch::Pair{Int, Int}; std::Number = 1)
    Fourier(first(ch), last(ch); std)
end

function initialparameters(rng::AbstractRNG, f::Fourier)
    modes = randn(rng, Float32, f.num_modes, f.in_dims) .* f.std
    return (modes = modes,)
end

function (f::Fourier)(x::AbstractVecOrMat, ps, st::NamedTuple)
    x = ps.modes * x
    x = 2 * eltype(x)(π) .* x
    return cat(sin.(x), cos.(x); dims = 1), st
end

function (f::Fourier)(x::AbstractArray, ps, st::NamedTuple)
    x = batched_mul(ps.modes, x)
    x = 2 * eltype(x)(π) .* x
    return cat(sin.(x), cos.(x); dims = 1), st
end

function Base.show(io::IO, f::Fourier)
    return print(io, "Fourier($(f.in_dims) => $(f.out_dims))")
end
