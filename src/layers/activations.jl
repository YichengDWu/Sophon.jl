@doc raw"""
    gaussian(x, a=0.2)
The Gaussian activation function.

```math
e^{\frac{- x^{2}}{2a^{2}}}
```
## Reference

[ramasinghe2021beyond](@cite)
"""
gaussian(x, a=0.2) = exp(-(x / NNlib.oftf(x, a))^2 / 2)
quadratic(x, a=5) = 1 / (1 + (NNlib.oftf(x, a) * x)^2)
multiquadratic(x, a=10) = 1 / sqrt((1 + (NNlib.oftf(x, a) * x)^2))
laplacian(x, a=0.01) = exp(-abs(x) / NNlib.oftf(x, a))
expsin(x, a=1) = exp(-sin(a * x))

@doc raw"""
    wu(x,a=1)

An activation function I designed for use in coordinate-mlps.

```math
\frac{x\left(5 x^{2}-1\right)}{\left(1+x^{2}\right)^{4}}
```
"""
function wu(x, a=1)
    x = NNlib.oftf(x, a) * x
    return x * (5 * x^2 - 1) / (1 + x^2)^4
end

@doc raw"""
    stan()

Self-scalable Tanh.

```math
\sigma(x^i) = tanh(x^i) + \beta^i * x^i * tanh(x^i)
```

## Reference

[gnanasambandam2022self](@cite)
"""
function stan end

kaiming_normal(::typeof(stan)) = Lux.glorot_uniform

function initialparameters(rng::AbstractRNG, d::Dense{true, typeof(stan)})
    return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
            bias=d.init_bias(rng, d.out_dims, 1),
            β=Lux.ones32(rng, d.out_dims, 1))
end

function initialparameters(rng::AbstractRNG, d::Dense{false, typeof(stan)})
    return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
            β=Lux.ones32(rng, d.out_dims, 1))
end

@inline function (d::Dense{false, typeof(stan)})(x::AbstractVecOrMat, ps, st::NamedTuple)
    z = ps.weight * x
    return tanh.(z) .* (1 .+ ps.β .* z), st
end

@inline function (d::Dense{true, typeof(stan)})(x::AbstractVector, ps, st::NamedTuple)
    z = ps.weight * x .+ vec(ps.bias)
    return tanh.(z) .* (1 .+ ps.β .* z), st
end

@inline function (d::Dense{true, typeof(stan)})(x::AbstractMatrix, ps, st::NamedTuple)
    z = ps.weight * x .+ ps.bias
    return tanh.(z) .* (1 .+ ps.β .* z), st
end
