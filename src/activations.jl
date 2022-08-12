gaussian(x, a=0.1f0) = exp(-(x / NNlib.oftf(x, a))^2 / 2)
quadratic(x, a=0.1f0) = 1 / (1 + (NNlib.oftf(x, a) * x)^2)
laplacian(x, a=0.1f0) = exp(-abs(x) / NNlib.oftf(x, a))
expsin(x, a=5f0) = exp(-sin(a * x))
