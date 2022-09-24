"""
    PINNAttention(H_net, U_net, V_net, fusion_layers)
    PINNAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                  hidden_dims::Int, num_layers::Int)

The output dimesion of `H_net` and the input dimension of `fusion_layers` must be the same.
For the second and the third constructor, `Dense` layers is used for `H_net`, `U_net`, and `V_net`.
Note that the first constructer does **not** contain the output layer, but the second one does.

```
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

## Reference

[wang2021understanding](@cite)
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
    FourierAttention(in_dims::Int, out_dims::Int, activation::Function, std;
                     hidden_dims::Int=512, num_layers::Int=6, modes::NTuple)
    FourierAttention(in_dims::Int, out_dims::Int, activation::Function, frequencies;
                     hidden_dims::Int=512, num_layers::Int=6, modes::NTuple)

A model that combines [`FourierFeatures`](@ref) and [`PINNAttention`](@ref).

```
x → [FourierFeature(x); x] → PINNAttention
```

## Arguments

  - `in_dims`: The input dimension.
    
      + `out_dims`: The output dimension.
      + `activation`: The activation function.
      + `std`: See [`FourierFeatures`](@ref).
      + `frequencies`: See [`FourierFeatures`](@ref).

## Keyword Arguments

  - `hidden_dim`: The hidden dimension of each hidden layer.
  - `num_layers`: The number of hidden layers.

## Examples

```julia
julia> FourierAttention(3, 1, sin, (1 => 10, 10 => 10, 50 => 10); hidden_dims=10, num_layers=3)
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
            layers = (layer_1 = Dense(10 => 10, sin), layer_2 = Dense(10 => 10, sin), layer_3 = Dense(10 => 10, sin), layer_4 = Dense(10 => 10, sin)),  # 440 parameters
        ),
    ),
    layer_3 = Dense(10 => 1),           # 11 parameters
)         # Total: 2_371 parameters,
          #        plus 90 states, summarysize 192 bytes
```
"""
function FourierAttention(in_dims::Int, out_dims::Int, activation::Function, freq;
                          hidden_dims::Int=512, num_layers::Int=6)
    fourierfeature = FourierFeature(in_dims, freq)
    encoder = SkipConnection(fourierfeature, vcat)
    attention_layers = PINNAttention(fourierfeature.out_dims + in_dims, out_dims,
                                     activation; hidden_dims=hidden_dims,
                                     num_layers=num_layers)
    return Chain(encoder, attention_layers)
end

"""
    FourierNet(ayer_sizes::NTuple, activation, modes::NTuple)

A model that combines [`FourierFeature`](@ref) and [`FullyConnected`](@ref).

```
x → FourierFeature → FullyConnected → y
```

# Arguments

  - `in_dims`: The number of input dimensions.
  - `layer_sizes`: A tuple of hidden dimensions used to construct `FullyConnected`.
  - `activation`: The activation function used to construct `FullyConnected`.
  - `modes`: A tuple of modes used to construct `FourierFeature`.

# Examples

```julia
julia> FourierNet((2, 30, 30, 1), sin, (1 => 10, 10 => 10, 50 => 10))
Chain(
    layer_1 = FourierFeature(2 => 60),
    layer_2 = Dense(60 => 30, sin),     # 1_830 parameters
    layer_3 = Dense(30 => 30, sin),     # 930 parameters
    layer_4 = Dense(30 => 1),           # 31 parameters
)         # Total: 2_791 parameters,
          #        plus 60 states, summarysize 112 bytes.

julia> FourierNet((2, 30, 30, 1), sin, (1, 2, 3, 4))
Chain(
    layer_1 = FourierFeature(2 => 16),
    layer_2 = Dense(16 => 30, sin),     # 510 parameters
    layer_3 = Dense(30 => 30, sin),     # 930 parameters
    layer_4 = Dense(30 => 1),           # 31 parameters
)         # Total: 1_471 parameters,
          #        plus 4 states, summarysize 96 bytes.
```
"""
function FourierNet(layer_sizes::NTuple{N, T}, activation::Function,
                    modes::NTuple) where {N, T <: Int}
    fourierfeature = FourierFeature(first(layer_sizes), modes)
    fc = FullyConnected((fourierfeature.out_dims, layer_sizes[2:end]...), activation)
    return Chain(fourierfeature, fc)
end

@doc raw"""
    Siren(in_dims::Int, out_dims::Int; hidden_dims::Int, num_layers::Int, omega=30.0f0,
          init_weight=nothing))
    Siren(layer_sizes::Int...; omega=30.0f0, init_weight=nothing)

Sinusoidal Representation Network.

## Keyword Arguments

  - `omega`: The `ω₀` used for the first layer.
  - `init_weight`: The initialization algorithm for the weights of the **input** layer. Note
    that all hidden layers use `kaiming_uniform` as the initialization algorithm. The default is
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

## Reference

[sitzmann2020implicit](@cite)

"""
function Siren(in_dims::Int, out_dims::Int; hidden_dims::Int, num_layers::Int, omega=30.0f0,
               init_weight::Union{Nothing, Function}=nothing)
    return _Siren((in_dims, ntuple(i -> hidden_dims, num_layers)..., out_dims), omega,
                  init_weight)
end

function Siren(layer_sizes::Int...; omega=30.0f0,
               init_weight::Union{Nothing, Function}=nothing)
    return _Siren(layer_sizes, omega, init_weight)
end

@generated function _Siren(layer_sizes::NTuple{N, T}, omega::O,
                           init_weight::W) where {N, T, O, W}
    layers = W === Nothing ? [:(Sine(layer_sizes[1] => layer_sizes[2]; omega=omega))] :
             [:(Dense(layer_sizes[1], layer_sizes[2]; init_weight=init_weight))]
    N == 2 && return layers[1]
    function get_layer(i)
        return :(Sine(layer_sizes[$i] => layer_sizes[$(i + 1)]))
    end
    append!(layers, [get_layer(i) for i in 2:(N - 2)])
    append!(layers,
            [
                :(Dense(layer_sizes[$(N - 1)] => layer_sizes[$N];
                        init_weight=kaiming_uniform(sin))),
            ])
    return :(Chain($(layers...)))
end

"""
    FullyConnected(layer_sizes::NTuple{N, Int}, activation; outermost = true,
                   init_weight = kaiming_uniform(activation),
                   init_bias = zeros32)
    FullyConnected(in_dims::Int, out_dims::Int, activation::Function;
                   hidden_dims::Int, num_layers::Int, outermost=true,
                   init_weight = kaiming_uniform(activation),
                   init_bias = zeros32)

Create fully connected layers.

## Arguments

  - `layer_sizes`: Number of dimensions of each layer.
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
function FullyConnected(layer_sizes::NTuple{N, T}, activation::Function;
                        outermost::Bool=true, init_bias=zeros32,
                        init_weight::Function=kaiming_uniform(activation)) where {N,
                                                                                  T <: Int}
    return FullyConnected(layer_sizes, activation, Val(outermost); init_weight=init_weight,
                          init_bias=init_bias)
end

function FullyConnected(in_dims::Int, out_dims::Int, activation::Function; hidden_dims::Int,
                        num_layers::Int, outermost::Bool=true,
                        init_weight::Function=kaiming_uniform(activation),
                        init_bias=zeros32)
    return FullyConnected((in_dims, ntuple(_ -> hidden_dims, num_layers)..., out_dims),
                          activation, Val(outermost); init_weight=init_weight,
                          init_bias=init_bias)
end

@generated function FullyConnected(layer_sizes::NTuple{N, T}, activation::Function,
                                   ::Val{F}; init_weight, init_bias) where {N, T <: Int, F}
    N == 2 &&
        return :(Dense(layer_sizes[1], layer_sizes[2], activation; init_weight=init_weight,
                       init_bias=init_bias))
    function get_layer(i)
        return :(Dense(layer_sizes[$i] => layer_sizes[$(i + 1)], activation;
                       init_weight=init_weight, init_bias=init_bias))
    end
    layers = [
        :(Dense(layer_sizes[1] => layer_sizes[2], activation; init_weight=init_weight,
                init_bias=init_bias)),
    ]
    append!(layers, [get_layer(i) for i in 2:(N - 2)])
    append!(layers,
            F ?
            [
                :(Dense(layer_sizes[$(N - 1)] => layer_sizes[$N]; init_weight=init_weight,
                        init_bias=init_bias)),
            ] : [get_layer(N - 1)])
    return :(Chain($(layers...)))
end

struct MultiplicativeFilterNet{F, L, O} <:
       AbstractExplicitContainerLayer{(:filters, :linear_layers, :output_layer)}
    filters::F
    linear_layers::L
    output_layer::O
end

function (m::MultiplicativeFilterNet)(x::AbstractArray, ps, st::NamedTuple)
    g, st_filter = m.filters(x, ps.filters, st.filters)
    z, st_linear = m.linear_layers(g, ps.linear_layers, st.linear_layers)
    y, st_output = m.output_layer(z, ps.output_layer, st.output_layer)

    st = (filters=st_filter, linear_layers=st_linear, output_layer=st_output)
    return y, st
end

@doc raw"""
    FourierFilterNet(in_dims::Int, out_dims::Int; hidden_dims::Int, num_layers::Int,
                     bandwidth::Real)

## Keyword Arguments
  - `bandwidth`: The maximum bandwidth of the network. The bandwidth is the sum of each filter's bandwidth.
## Parameters

- Parameters of the filters:
```math
    W\sim \mathcal{U}(-\frac{ω}{n}, \frac{ω}{n}), \quad b\sim \mathcal{U}(-\pi, \pi),
```
  where `n` is the number of filters.

  For a periodic function with period ``P``, the Fourier series in amplitude-phase form is
```math
s_N(x)=\frac{a_0}{2}+\sum_{n=1}^N{a_n}\cdot \sin \left( \frac{2\pi}{P}nx+\varphi _n \right)
```
  We have the following relation between the banthwidth and the parameters of the model:
```math
ω = 2πB=\frac{2πN}{P}.
```
  where ``B`` is the bandwidth of the network.

## References

[fathony2021multiplicative](@cite)

[lindell2021bacon](@cite)
"""
function FourierFilterNet(in_dims::Int, out_dims::Int; hidden_dims::Int, num_layers::Int,
                          bandwidth::Real)
    names = ntuple(i -> Symbol("filter_$i"), num_layers)
    scale = 2.0f0π * bandwidth / num_layers
    layers = ntuple(i -> Dense(in_dims, hidden_dims, sin; init_bias=init_uniform(1.0f0π),
                               init_weight=init_uniform(scale)), num_layers)
    nt = NamedTuple{names}(layers)
    filters = BranchLayer{typeof(nt)}(nt)

    layers = ntuple(i -> Dense(hidden_dims, hidden_dims; init_weight=kaiming_uniform(sin)),
                    num_layers - 1)
    linear_layers = PairwiseFusion(.*, layers...)

    output_layer = Dense(hidden_dims, out_dims; init_weight=kaiming_uniform(sin))
    return MultiplicativeFilterNet{typeof(filters), typeof(linear_layers),
                                   typeof(output_layer)}(filters, linear_layers,
                                                         output_layer)
end

"""
    BACON(in_dims::Int, out_dims::Int, N::Int, period::Real; hidden_dims::Int, num_layers::Int)

Band-limited Coordinate Networks (BACON) from [lindell2021bacon](@cite). Similar to [`FourierFilterNet`](@ref) but the
frequcies are dicrete and nontrainable.

Tips: It is recommended to set `period` to be `1,2,π` or `2π` for better performance.
"""
function BACON(in_dims::Int, out_dims::Int, N::Int, period::Real; hidden_dims::Int,
               num_layers::Int)
    names = ntuple(i -> Symbol("filter_$i"), num_layers)
    Ns = ntuple(_ -> N ÷ num_layers, num_layers)
    if N % num_layers != 0
        Ns = (Ns[1:(end - 1)]..., Ns[end] + N % num_layers)
    end

    layers = ntuple(i -> DiscreteFourierFeature(in_dims, hidden_dims, Ns[i], period),
                    num_layers)
    nt = NamedTuple{names}(layers)
    filters = BranchLayer{typeof(nt)}(nt)

    layers = ntuple(i -> Dense(hidden_dims, hidden_dims; init_weight=kaiming_uniform(sin)),
                    num_layers - 1)
    linear_layers = PairwiseFusion(.*, layers...)

    output_layer = Dense(hidden_dims, out_dims; init_weight=kaiming_uniform(sin))
    return MultiplicativeFilterNet{typeof(filters), typeof(linear_layers),
                                   typeof(output_layer)}(filters, linear_layers,
                                                         output_layer)
end
