@inline GPUComponentArray64(nt::NamedTuple) = nt |> ComponentArray |> gpu .|> Float64
@inline GPUComponentArray(nt::NamedTuple) = nt |> ComponentArray |> gpu
@inline ComponentArray64(nt::NamedTuple) = nt |> ComponentArray .|> Float64

"""
    kaiming_uniform(rng::AbstractRNG, size...; gain = √2f0)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=√2.0f0)
    bound = Float32(√3.0f0 * gain / sqrt(first(Lux._nfan(dims...))))
    return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

function kaiming_uniform(nonlinearity::Union{Type{<:Function}, Function})
    return (rng::AbstractRNG, dims::Integer...) -> kaiming_uniform(rng, dims...;
                                                                   gain=calculate_gain(nonlinearity))
end

ChainRulesCore.@non_differentiable kaiming_uniform(::Any...)

"""
    kaiming_normal(rng::AbstractRNG, size...; gain = √2f0)

Return an `Array{Float32}` of the given `size` containing random numbers taken from a normal
distribution standard deviation `gain / sqrt(fan_in)`

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, dims::Integer...; gain::Real=√2.0f0)
    std = Float32(gain / sqrt(first(Lux._nfan(dims...))))
    return randn(rng, Float32, dims...) .* std
end

function kaiming_normal(nonlinearity::Union{Type{<:Function}, Function})
    return (rng::AbstractRNG, dims::Integer...) -> kaiming_normal(rng, dims...;
                                                                  gain=calculate_gain(nonlinearity))
end

ChainRulesCore.@non_differentiable kaiming_normal(::Any...)

@inline calculate_gain(::typeof(relu)) = √2.0f0
@inline calculate_gain(::typeof(tanh)) = Float32(5 / 3)
@inline calculate_gain(::typeof(sigmoid)) = 1
@inline calculate_gain(::typeof(identity)) = 1
@inline calculate_gain(::typeof(sin)) = √2.0f0 # Siren
@inline calculate_gain(::Type{<:Function}) = 1 # default
@inline calculate_gain(f::Function) = calculate_gain(typeof(f))

"""
    init_uniform(rng::AbstractRNG, size...; scale = 1)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from the
uniform distribution on the interval `[-scale, scale]`.
"""
@inline function init_uniform(rng::AbstractRNG, dims::Integer...; scale::Real=1)
    return (rand(rng, Float32, dims...) .- 0.5f0) .* 2scale
end

@inline function init_uniform(scale::Real)
    return (rng::AbstractRNG, dims::Integer...) -> init_uniform(rng, dims...; scale=scale)
end

ChainRulesCore.@non_differentiable init_uniform(::Any...)

"""
    init_normal(rng::AbstractRNG, size...; std = 1)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from the
standard normal distribution.
"""
@inline function init_normal(rng::AbstractRNG, dims::Integer...; std::Real=1)
    return randn(rng, Float32, dims...) .* std
end

@inline function init_normal(std::Real)
    return (rng::AbstractRNG, dims::Integer...) -> init_normal(rng, dims...; std=std)
end

ChainRulesCore.@non_differentiable init_normal(::Any...)

@memoize LRU{Tuple{AbstractArray,AbstractArray},AbstractArray}(maxsize=1000) function cached_sinWx(W, x)
    return sin.(W * x)
end

@memoize LRU{Tuple{AbstractArray,AbstractArray},AbstractArray}(maxsize=1000) function cached_sinpiWx(W, x)
    return sinpi.(W * x)
end

@memoize LRU{Tuple{AbstractArray,AbstractArray},AbstractArray}(maxsize=1000) function cached_cosWx(W, x)
    return cos.(W * x)
end

@memoize LRU{Tuple{AbstractArray,AbstractArray},AbstractArray}(maxsize=1000) function cached_cospiWx(W, x)
    return cospi.(W * x)
end

function ChainRulesCore.rrule(::typeof(cached_sinWx), W::AbstractMatrix,
                              x::AbstractVecOrMat)
    y = cached_sinWx(W, x)
    function cached_sinWx_pullback(ȳ)
        f̄ = NoTangent()
        W̄ = @thunk((ȳ .* cached_cosWx(W, x))*x')
        x̄ = W' * (cached_cosWx(W, x) .* ȳ)
        return f̄, W̄, x̄
    end
    return y, cached_sinWx_pullback
end

function ChainRulesCore.rrule(::typeof(cached_sinpiWx), W::AbstractMatrix,
                              x::AbstractVecOrMat)
    y = cached_sinpiWx(W, x)
    function cached_sinWx_pullback(ȳ)
        f̄ = NoTangent()
        W̄ = @thunk((ȳ .* π .* cached_cospiWx(W, x))*x')
        x̄ = W' * (π .* cached_cospiWx(W, x) .* ȳ)
        return f̄, W̄, x̄
    end
    return y, cached_sinWx_pullback
end

function ChainRulesCore.rrule(::typeof(cached_cosWx), W::AbstractMatrix,
                              x::AbstractVecOrMat)
    y = cached_cosWx(W, x)
    function cached_cosWx_pullback(ȳ)
        f̄ = NoTangent()
        W̄ = @thunk((ȳ .* -cached_sinWx(W, x))*x')
        x̄ = W' * (-cached_sinWx(W, x) .* ȳ)
        return f̄, W̄, x̄
    end
    return y, cached_cosWx_pullback
end

function ChainRulesCore.rrule(::typeof(cached_cospiWx), W::AbstractMatrix,
                              x::AbstractVecOrMat)
    y = cached_cospiWx(W, x)
    function cached_cospiWx_pullback(ȳ)
        f̄ = NoTangent()
        W̄ = @thunk((ȳ .* π .* -cached_sinpiWx(W, x))*x')
        x̄ = W' * (π .* -cached_sinpiWx(W, x) .* ȳ)
        return f̄, W̄, x̄
    end
    return y, cached_cospiWx_pullback
end
