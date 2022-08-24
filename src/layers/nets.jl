"""
    PINNAttention(H_net, U_net, V_net, fusion_layers)
    PINNAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                  hidden_dims::Int, num_layers::Int)

The output dimesion of `H_net` and the input dimension of `fusion_layers` must be the same.
For the second and the third constructor, `Dense` layers is used for `H_net`, `U_net`, and `V_net`.
Note that the first constructer does not contain the output layer.

```julia
                 x → U_net → u                           u
                               ↘                           ↘
x → H_net →  h1 → fusionlayer1 → connection → fusionlayer2 → connection
                               ↗                           ↗
                 x → V_net → v                           v
```

## Arguments

  - `H_net`: `AbstractExplicitLayer`.
  - `U_net`: `AbstractExplicitLayer`.
  - `V_net`: `AbstractExplicitLayer`.
  - `fusion_layers`: `Chain`.

## Keyword Arguments

  - `num_layers`: The number of hidden layers.
  - `hidden_dims`: The number of hidden dimensions of each hidden layer.

## References

[1] Wang, Sifan, Yujun Teng, and Paris Perdikaris. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081
"""
struct PINNAttention{H, U, V, F <: TriplewiseFusion} <:
       AbstractExplicitContainerLayer{(:H_net, :U_net, :V_net, :fusion)}
    H_net::H
    U_net::U
    V_net::V
    fusion::F
end

function PINNAttention(H_net::AbstractExplicitLayer, U_net::AbstractExplicitLayer,
                       V_net::AbstractExplicitLayer, fusion_layers::AbstractExplicitLayer)
    fusion = TriplewiseFusion(attention_connection, fusion_layers)
    return PINNAttention{typeof(H_net), typeof(U_net), typeof(V_net), typeof(fusion)}(H_net,
                                                                                      U_net,
                                                                                      V_net,
                                                                                      fusion)
end

function PINNAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                       hidden_dims::Int, num_layers::Int)
    H_net = Dense(in_dims, hidden_dims, activation)
    U_net = Dense(in_dims, hidden_dims, activation)
    V_net = Dense(in_dims, hidden_dims, activation)
    fusion_layers = FullyConnected(hidden_dims, hidden_dims, activation;
                                   hidden_dims=hidden_dims, num_layers=num_layers,
                                   outermost=false)
    return Chain(PINNAttention(H_net, U_net, V_net, fusion_layers),
                 Dense(hidden_dims, out_dims))
end

function (m::PINNAttention)(x::AbstractArray, ps, st::NamedTuple)
    H, st_h = m.H_net(x, ps.H_net, st.H_net)
    U, st_u = m.U_net(x, ps.U_net, st.U_net)
    V, st_v = m.V_net(x, ps.V_net, st.V_net)
    F, st_f = m.fusion((H, U, V), ps.fusion, st.fusion)
    st = merge(st, (U_net=st_u, V_net=st_v, H_net=st_h, fusion=st_f))
    return F, st
end

attention_connection(z, u, v) = (1 .- z) .* u .+ z .* v

"""
    FourierAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                     hidden_dims::Int=512, num_layers::Int=6, modes::NTuple)

```
x → [FourierFeature(x); x] → PINNAttention
```

## Arguments

  - `in_dims`: The input dimension.

## Keyword Arguments

  - `modes`: A tuple of pairs of random frequencies and the number of samples.
  - `hidden_dim`: The hidden dimension of each hidden layer.
  - `num_layers`: The number of hidden layers.

## Examples

```julia
julia> FourierAttention(3, 1, sin; hidden_dims=10, num_layers=3,
                        modes=(1 => 10, 10 => 10, 50 => 10))
Chain(
    layer_1 = SkipConnection(
        FourierFeature(3 => 60),
        vcat
    ),
    layer_2 = PINNAttention(
        H_net = Dense(63 => 10, sin),   # 640 parameters
        U_net = Dense(63 => 10, sin),   # 640 parameters
        V_net = Dense(63 => 10, sin),   # 640 parameters
        fusion = TriplewiseFusion(
            layers = (layer_1 = Dense(10 => 10, sin), layer_2 = Dense(10 => 10, sin), layer_3 = Dense(10 => 10, sin), layer_4 = Dense(10 => 1)),  # 341 parameters
        ),
    ),
)         # Total: 2_261 parameters,
          #        plus 90 states, summarysize 176 bytes.
```
"""
function FourierAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                          hidden_dims::Int=512, num_layers::Int=6, modes::NTuple)
    fourierfeature = FourierFeature(in_dims, modes)
    encoder = SkipConnection(fourierfeature, vcat)
    attention_layers = PINNAttention(fourierfeature.out_dims + in_dims, out_dims,
                                     activation; hidden_dims=hidden_dims,
                                     num_layers=num_layers)
    return Chain(encoder, attention_layers)
end

"""
    SirenAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
    hidden_dims::Int=512, num_layers::Int=6, omega=30.0f0))

```julia
x → Dense(..., activation) → u                           u
                              ↘                           ↘
x → Siren[1] →  s1 → Siren[2] → connection → fusionlayer2 → connection
                              ↗                           ↗
x → Dense(..., activation) → v                           v
```

Combined model of [`Siren`](@ref) and [`PINNAttention`](@ref).
"""
function SirenAttention(in_dims::Int, out_dims::Int, activation::Function=relu;
                        hidden_dims::Int=512, num_layers::Int=6, omega=30.0f0)

    U_net = Dense(in_dims, hidden_dims, activation)
    V_net = Dense(in_dims, hidden_dims, activation)
    siren = Siren(in_dims, out_dims; hidden_dims=hidden_dims,
                  num_layers=num_layers, omega=omega)
    H_net = siren[1]
    fusion_layers = Chain(Tuple(siren.layers)[2:end])

    attention_layers = PINNAttention(H_net, U_net, V_net, fusion_layers)
    return attention_layers
end

"""
    MultiscaleFourier(in_dims::Int, layer_dims::NTuple, activation=identity, modes::NTuple)

Multi-scale Fourier Feature Net.

```
x → FourierFeature → FullyConnected → y
```

# Arguments

  - `in_dims`: The number of input dimensions.
  - `layer_dims`: A tuple of hidden dimensions used to construct `FullyConnected`.
  - `activation`: The activation function used to construct `FullyConnected`.
  - `modes`: A tuple of modes used to construct `FourierFeature`.

# Keyword Arguments

  - `modes`: A tuple of modes used to construct `FourierFeature`.

# Examples

```julia
julia> m = MultiscaleFourier(2, (30, 30, 1), sin; modes=(1 => 10, 10 => 10, 50 => 10))
Chain(
    layer_1 = FourierFeature(2 => 60),
    layer_2 = Dense(60 => 30, sin),     # 1_830 parameters
    layer_3 = Dense(30 => 30, sin),     # 930 parameters
    layer_4 = Dense(30 => 1),           # 31 parameters
)         # Total: 2_791 parameters,
          #        plus 60 states, summarysize 112 bytes.
```

# References

[1] Wang, Sifan, Hanwen Wang, and Paris Perdikaris. “On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks.” Computer Methods in Applied Mechanics and Engineering 384 (2021): 113938.
"""
function MultiscaleFourier(in_dims::Int,
                           out_dims::NTuple{N1, Int}=(ntuple(i -> 512, 6)..., 1),
                           activation::Function=sin;
                           modes::NTuple{N2, Pair{S, Int}}=(1 => 64, 10 => 64, 20 => 64,
                                                            50 => 32, 100 => 32)) where {N1,
                                                                                         N2,
                                                                                         S}
    fourierfeature = FourierFeature(in_dims, modes)
    fc = FullyConnected((fourierfeature.out_dims, out_dims...), activation)
    return Chain(fourierfeature, fc)
end

@doc raw"""
    Siren(in_dims::Int, out_dims::Int; hidden_dims::Int, num_layers::Int, omega=30.0f0,
          init_weight=nothing))
    Siren(layer_dims::Int...; omega=30.0f0, init_weight=nothing)

Sinusoidal Representation Network.

## Keyword Arguments

  - `omega`: The `ω₀` used for the first layer.
  - `init_weight`: The initialization algorithm for the weights of the **input** layer. Note
    that all hidden layers use `kaiming_uniform` as the initialization algorithm. If not
    specified, the default is
    ```math
        W\sim \mathcal{U}\left(-\frac{\omega}{fan_{in}}, \frac{\omega}{fan_{in}}\right)
    ```

## Examples

```julia
julia> Siren(2, 32, 32, 1; omega=5.0f0)
Chain(
    layer_1 = Dense(2 => 32, sin),      # 96 parameters
    layer_2 = Dense(32 => 32, sin),     # 1_056 parameters
    layer_3 = Dense(32 => 1),           # 33 parameters
)         # Total: 1_185 parameters,
          #        plus 0 states, summarysize 48 bytes.

julia> Siren(3, 1; hidden_dims=20, num_layers=3)
Chain(
    layer_1 = Dense(3 => 20, sin),      # 80 parameters
    layer_2 = Dense(20 => 20, sin),     # 420 parameters
    layer_3 = Dense(20 => 20, sin),     # 420 parameters
    layer_4 = Dense(20 => 1),           # 21 parameters
)         # Total: 941 parameters,
          #        plus 0 states, summarysize 64 bytes.

# Use your own initialization algorithm for the input layer.
julia> init_weight(rng::AbstractRNG, out_dims::Int, in_dims::Int) = randn(rng, Float32, out_dims, in_dims) .* 2.5f0
julia> chain = Siren(2, 1; num_layers = 4, hidden_dims = 50, init_weight = init_weight)
```

## References

[1] Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in Neural Information Processing Systems 33 (2020): 7462-7473.
"""
function Siren(in_dims::Int, out_dims::Int; hidden_dims::Int, num_layers::Int, omega=30.0f0,
               init_weight::Union{Nothing, Function}=nothing)
    return _Siren((in_dims, ntuple(i -> hidden_dims, num_layers)..., out_dims), omega,
                  init_weight)
end

function Siren(layer_dims::Int...; omega=30.0f0,
               init_weight::Union{Nothing, Function}=nothing)
    return _Siren(layer_dims, omega, init_weight)
end

@generated function _Siren(layer_dims::NTuple{N, T}, omega::O,
                           init_weight::W) where {N, T, O, W}
    layers = W === Nothing ? [:(Sine(layer_dims[1] => layer_dims[2]; omega=omega))] :
             [:(Dense(layer_dims[1], layer_dims[2]; init_weight=init_weight))]
    N == 2 && return layers[1]
    function get_layer(i)
        return :(Sine(layer_dims[$i] => layer_dims[$(i + 1)]))
    end
    append!(layers, [get_layer(i) for i in 2:(N - 2)])
    append!(layers,
            [
                :(Dense(layer_dims[$(N - 1)] => layer_dims[$N];
                        init_weight=kaiming_uniform(sin))),
            ])
    return :(Chain($(layers...)))
end

"""
    FullyConnected(layer_dims::NTuple{N, Int}, activation; outermost = true, init_weight = kaiming_uniform(activation))
    FullyConnected(in_dims::Int, out_dims::Int, activation::Function;
                   hidden_dims::Int, num_layers::Int, outermost=true, init_weight = kaiming_uniform(activation))

Create fully connected layers.

## Arguments

  - `layer_dims`: Number of dimensions of each layer.
  - `hidden_dims`: Number of hidden dimensions.
  - `num_layers`: Number of layers.
  - `activation`: Activation function.

## Keyword Arguments

  - `outermost`: Whether to use activation function for the last layer. If `false`, the activation function is applied
    to the output of the last layer.
  - `init_weight`: Initialization method for the weights.

## Example

```julia
julia> fc = FullyConnected((1, 12, 24, 32), relu)
Chain(
    layer_1 = Dense(1 => 12, relu),     # 24 parameters
    layer_2 = Dense(12 => 24, relu),    # 312 parameters
    layer_3 = Dense(24 => 32),          # 800 parameters
)         # Total: 1_136 parameters,
          #        plus 0 states, summarysize 48 bytes.

julia> fc = FullyConnected(1, 10, relu; hidden_dims=20, num_layers=3)
Chain(
    layer_1 = Dense(1 => 20, relu),     # 40 parameters
    layer_2 = Dense(20 => 20, relu),    # 420 parameters
    layer_3 = Dense(20 => 20, relu),    # 420 parameters
    layer_4 = Dense(20 => 10),          # 210 parameters
)         # Total: 1_090 parameters,
          #        plus 0 states, summarysize 64 bytes.
```
"""
function FullyConnected(layer_dims::NTuple{N, T}, activation::Function;
                        outermost::Bool=true,
                        init_weight::Function=kaiming_uniform(activation)) where {N,
                                                                                  T <: Int}
    return FullyConnected(layer_dims, activation, Val(outermost); init_weight=init_weight)
end

function FullyConnected(in_dims::Int, out_dims::Int, activation::Function; hidden_dims::Int,
                        num_layers::Int, outermost::Bool=true,
                        init_weight::Function=kaiming_uniform(activation))
    return FullyConnected((in_dims, ntuple(_ -> hidden_dims, num_layers)..., out_dims),
                          activation, Val(outermost); init_weight=init_weight)
end

@generated function FullyConnected(layer_dims::NTuple{N, T}, activation::Function, ::Val{F};
                                   init_weight) where {N, T <: Int, F}
    N == 2 &&
        return :(Dense(layer_dims[1], layer_dims[2], activation; init_weight=init_weight))
    function get_layer(i)
        return :(Dense(layer_dims[$i] => layer_dims[$(i + 1)], activation;
                       init_weight=init_weight))
    end
    layers = [:(Dense(layer_dims[1] => layer_dims[2], activation; init_weight=init_weight))]
    append!(layers, [get_layer(i) for i in 2:(N - 2)])
    append!(layers,
            F ?
            [:(Dense(layer_dims[$(N - 1)] => layer_dims[$N]; init_weight=init_weight))] :
            [get_layer(N - 1)])
    return :(Chain($(layers...)))
end
