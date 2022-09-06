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
quadratic(x, a=1) = 1 / (1 + (NNlib.oftf(x, a) * x)^2)
multiquadratic(x, a=10) = 1 / sqrt((1 + (NNlib.oftf(x, a) * x)^2))
laplacian(x, a=0.01) = exp(-abs(x) / NNlib.oftf(x, a))
expsin(x, a=1) = xp(-sin(a * x))

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
