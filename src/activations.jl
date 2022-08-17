gaussian(x, a=0.02) = exp(-(x / a)^2 / 2)
quadratic(x, a=1) = 1 / (1 + (NNlib.oftf(x, a) * x)^2)
laplacian(x, a=1) = exp(-abs(x) / NNlib.oftf(x, a))
expsin(x, a=1) = exp(-sin(a * x))
