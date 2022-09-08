# Fitting a nonlinear discontinuous function

This example is taken from [here](https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2020.0334). However, we do not use adaptive activation functions. Instead, we show that using suitable non-parametric activation functions immediately performs better.


Consider the following  discontinuous  function  with  discontinuity  at ``x=0``:

```math
u(x)= \begin{cases}0.2 \sin (18 x) & \text { if } x \leq 0 \\ 1+0.3 x \cos (54 x) & \text { otherwise }\end{cases}
```
The domain is ``[-1,1]``. The number of training points used is `50`.

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

function generate_data(n=50)
    x = reshape(collect(range(-1.0f0, 1.0f0, n)), (1, n))
    y = u.(x)
    return (x, y)
end
```

Let's visualize the data.

```@example ds
x_train, y_train = generate_data(50)
x_test, y_test = generate_data(200)
Plots.plot(vec(x_test), vec(y_test),label=false)
savefig("u.svg"); nothing # hide
```
![](u.svg)

## Naive Neural Nets

First we demonstrate show naive fully connected neural nets could be really bad at fitting this function.
```@example ds
model = FullyConnected((1,50,50,50,50,1), relu)
```
### Train the model

```@example ds
function train(model, x, y)
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
### Plot the result
```@example ds
@time ps, st = train(model, x_train, y_train)
y_pred = model(x_test,ps,st)[1]
Plots.plot(vec(x_test), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x_test), vec(y_test),label="Exact",legend=:topleft)
savefig("result1.svg"); nothing # hide
```
![](result1.svg)

## Siren
We use four hidden layers with 50 neurons in each.
```@example ds
model = Siren(1,50,50,50,50,1; omega = 30f0)
```
```@example ds
@time ps, st = train(model, x_train, y_train)
y_pred = model(x_test,ps,st)[1]
Plots.plot(vec(x_test), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x_test), vec(y_test),label="Exact",legend=:topleft)
savefig("result.svg"); nothing # hide
```

![](result.svg)

As we can see the model overfits the data, and the high frequencies cannot be optimized away. We need to tunning the hyperparameter `omega`

```@example ds
model = Siren(1,50,50,50,50,1; omega = 10f0)
```
```@example ds
@time ps, st = train(model, x_train, y_train)
y_pred = model(x_test,ps,st)[1]
Plots.plot(vec(x_test), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x_test), vec(y_test),label="Exact",legend=:topleft)
savefig("result10.svg"); nothing # hide
```
![](result10.svg)

## Gaussian activation function

We can also try using a fully connected net with the [`gaussian`](@ref) activation function.

```@example ds
model = FullyConnected((1,50,50,50,50,1), gaussian)
```

```@example ds
@time ps, st = train(model, x_train, y_train)
y_pred = model(x_test,ps,st)[1]
Plots.plot(vec(x_test), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x_test), vec(y_test),label="Exact",legend=:topleft)
savefig("result2.svg"); nothing # hide
```
![](result2.svg)


## Quadratic activation function

[`quadratic`](@ref) is much cheaper to compute compared to the Gaussain activation function.


```@example ds
model = FullyConnected((1,50,50,50,50,1), quadratic)
```

```@example ds
@time ps, st = train(model, x_train, y_train)
y_pred = model(x_test,ps,st)[1]
Plots.plot(vec(x_test), vec(y_pred),label="Prediction",line = (:dot, 4))
Plots.plot!(vec(x_test), vec(y_test),label="Exact",legend=:topleft)
savefig("result3.svg"); nothing # hide
```
![](result3.svg)

## Conclusion

The "neural network suppresses high frequency components" is a misrepresentation of spectral bias. The accurate way of putting it is that the lower frequencies in the error are optimized first in the optimization process. This can be seen in Siren's example of overfitting data, where you do not have implicit regularization.

Mainstream attributes the phenomenon that neural networks "suppress" high frequencies to gradient descent. This is not the whole picture. Initialization also plays an important role. Siren mitigats this problem by initializing larger weights in the first layer, while activation functions such as gassian have large enough gradients and sufficiently large support of the second derivative with proper parameters. Please refer to [sitzmann2020implicit](@cite), [ramasinghe2021beyond](@cite) and [ramasinghe2022regularizing](@cite) if you want to dive deeper into this.
