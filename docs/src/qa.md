## Q: How can I train the model using GPUs?

A: To train the model on GPUs, invoke the gpu function on instances of PINN:

```julia
using Lux
pinn = gpu(PINN(...))
```
## Q: How can I monitor the loss for each loss function?

A: Data points are stored in `prob.p`. Call `Sophon.residual_function_x` with the corresponding arguments to obtain the residual of each data point:
```julia
residual = Sophon.residual_function_1(prob.p[1], res.u)
```

If you want to monitor the loss during training, create a callback function like the following:
```julia
function callback(p, _)
    loss = sum(abs2, Sophon.residual_function_1(prob.p[1], p))
    println("loss: $loss")
    return false
end
```

Finally, pass the callback function to `Optimization.solve` to monitor the loss as the training progresses.

## Q: How can I inspect the generated symbolic loss function?

A: Simply replace `Sophon.discretize` by `Sophon.symbolic_discretize`.
