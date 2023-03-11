```@meta
CurrentModule = Sophon
```

# Sophon

Sophon.jl is a Julia package for solving partial differential equations (PDEs) using physics-informed neural networks (PINNs). We will go through the steps to solve the nonlinear Schrödinger equation using Sophon.jl.

## Step 1: Import the required packages
```julia
using ModelingToolkit, IntervalSets, Sophon, CairoMakie
using Optimization, OptimizationOptimJL
```
## Step 2: Define the PDE system

In this step, we define the PDE system for the nonlinear Schrödinger equation using ModelingToolkit.jl.

```julia
@parameters x,t
@variables u(..), v(..)
Dₜ = Differential(t)
Dₓ² = Differential(x)^2

eqs=[Dₜ(u(x,t)) ~ -Dₓ²(v(x,t))/2 - (abs2(v(x,t)) + abs2(u(x,t))) * v(x,t),
     Dₜ(v(x,t)) ~  Dₓ²(u(x,t))/2 + (abs2(v(x,t)) + abs2(u(x,t))) * u(x,t)]

bcs = [u(x, 0.0) ~ 2sech(x),
       v(x, 0.0) ~ 0.0,
       u(-5.0, t) ~ u(5.0, t),
       v(-5.0, t) ~ v(5.0, t)]

domains = [x ∈ Interval(-5.0, 5.0),
           t ∈ Interval(0.0, π/2)]

@named pde_system = PDESystem(eqs, bcs, domains, [x,t], [u(x,t),v(x,t)])
```

The `@parameters` macro defines the parameters of the PDE system, and the `@variables` macro defines the dependent variables. We use `Differential` to define the derivatives with respect to time and space. The `eqs` array defines the equations in the PDE system. The bcs array defines the boundary conditions. The `domains` array defines the spatial and temporal domains of the PDE system. Finally, we use the `@named` macro to give a name to the PDE system.

## Step 3: Define the neural network architecture
Next, we define the physics-informed neural network (PINN) using Sophon.jl. In this example, we will use a Siren network with 2 sine layers and 1 cosine layer for each variable, and 16 hidden dimensions per layer. We will use 4 layers for both variables, and set the frequency parameter $\omega$ to 1.0.

```julia
pinn = PINN(u = Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0),
            v = Siren(2,1; hidden_dims=16,num_layers=4, omega = 1.0))
            
sampler = QuasiRandomSampler(500, (200,200,20,20))
strategy = NonAdaptiveTraining(1,(10,10,1,1))
```
We define a physics-informed neural network (PINN) with the pinn variable. The PINN macro takes a dictionary that maps the dependent variables to their corresponding neural network architecture. In this case, we use the Siren architecture for both u and v with 2 input dimensions, 1 output dimension, 16 hidden dimensions, and 4 layers. We also set the frequency of the sine activation functions to 1.0.

## Step 4: Create a QuasiRandomSampler object
Here, we create a QuasiRandomSampler object with 500 sample points, where the first argument corresponds to the number of data points for each equation, and the second 
argument corresponds to the number of data points for each boundary condition.

## Step 5: Define a training strategy
Here, we use a NonAdaptiveTraining strategy with `1` as the weight of all equations, and `(10,10,1,1)` for the four boundary conditions.

## Step 6: Discretize the PDE system using Sophon
```julia
prob = Sophon.discretize(pde_system, pinn, sampler, strategy)
```
## Step 7: Solve the optimization problem
```julia
res = Optimization.solve(prob, BFGS(); maxiters=2000)
```

```@index
```

```@autodocs
Modules = [Sophon]
Private = false
```
