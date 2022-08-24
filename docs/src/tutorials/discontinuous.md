# Fitting a nonlinear discontinuous function

This example is taken from [here](https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2020.0334). However, we do not use adaptive activation functions. Instead, we show that using suitable non-parametric activation functions immediately performs better.


The following  discontinuous  function  with  discontinuity  at ``x=0``  location  is approximated by [`Siren`](@ref).

```math
u(x)= \begin{cases}0.2 \sin (18 x) & \text { if } x \leq 0 \\ 1+0.3 x \cos (54 x) & \text { otherwise }\end{cases}
```
The domain is ``[-1,1]``. The number of training points used is `300`.

## Import pacakges
```@example ds
using Lux, Sophon
using NNlib, Optimisers, Plots, Random, Statistics, Zygote
```

## Dataset

```@example ds
function u(x)
    if x <= 0
        return 0.2 * sin(18 * x)
    else
        return 1 + 0.3 * x * cos(54 * x)
    end
end

function generate_data(n=300)
    x = reshape(collect(range(-1.0f0, 1.0f0, n)), (1, n))
    y = u.(x)
    return (x, y)
end
```

Let's visualize the data.

```@example ds
x, y = generate_data()
Plots.plot(vec(x), vec(y),label=false)
savefig("u.svg"); nothing # hide
```
![](u.svg)

## Model
We use four hidden layers with 50 neurons in each.
```@example ds
model = Siren(1,50,50,50,50,1)
```
## Train the model

```@example ds
function train(model)
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Adam()
    st_opt = Optimisers.setup(opt,ps)
    function loss(model, ps, st, x, y)
        y_pred, _ = model(x, ps, st)
        mes = mean(abs2, y_pred .- y)
        return mes
    end

    for i in 1:2000
        gs = gradient(p->loss(model,p,st,x,y), ps)[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
        if i % 100 == 1 || i == 2000
            println("Epoch $i ||  ", loss(model,ps,st,x,y))
        end
    end
    return ps, st
end

```
## Results
```@example ds
@time ps, st = train(model)
y_pred = model(x,ps,st)[1]
Plots.plot(vec(x), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x), vec(y),label="Exact",legend=:topleft)
savefig("result.svg"); nothing # hide
```

![](result.svg)

## Gaussian activation function

We can also try using a fully connected net with the [`gaussian`](@ref) activation function.

```@example ds
model = FullyConnected((1,50,50,50,50,1), gaussian)
```

```@example ds
@time ps, st = train(model)
y_pred = model(x,ps,st)[1]
Plots.plot(vec(x), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x), vec(y),label="Exact",legend=:topleft)
savefig("result2.svg"); nothing # hide
```
![](result2.svg)


## Quadratic activation function

[`quadratic`](@ref) is much cheaper to compute compared to the Gaussain activation function.


```@example ds
model = FullyConnected((1,50,50,50,50,1), quadratic)
```

```@example ds
@time ps, st = train(model)
y_pred = model(x,ps,st)[1]
Plots.plot(vec(x), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x), vec(y),label="Exact",legend=:topleft)
savefig("result3.svg"); nothing # hide
```
![](result3.svg)
