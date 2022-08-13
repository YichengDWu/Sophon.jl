"""
    PINNAttention(H_net, U_net, V_net, fusion_layers)
    PINNAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                  hidden_dims::Int, num_layers::Int)

The output dimesion of `H_net` and the input dimension of `fusion_layers` must be the same.
For the second and the third constructor, `Dense` layers is used for `H_net`, `U_net`, and `V_net`.
Note that the first constructer does not contain the output layer.

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
    hidden_dims::Int=512, num_layers::Int=6)

```
x -> Sine -> PINNAttention
```
"""
function SirenAttention(in_dims::Int, out_dims::Int, activation::Function=sin;
                        hidden_dims::Int=512, num_layers::Int=6, omega = 30f0)
    sine = Sine(in_dims, hidden_dims[1]; is_first=true, omega = omega)
    attention_layers = PINNAttention(hidden_dims[1], out_dims,
                                     activation; hidden_dims=hidden_dims,
                                     num_layers=num_layers - 1)
    return Chain(sine, attention_layers)
end

"""
    MultiscaleFourier(in_dims::Int, out_dims::NTuple, activation=identity, modes::NTuple)

Multi-scale Fourier Feature Net.

```
x → FourierFeature → FullyConnected → y
```

# Arguments

  - `in_dims`: The number of input dimensions.
  - `hidden_dims`: A tuple of hidden dimensions used to construct `FullyConnected`.
  - `activation`: The activation function used to construct `FullyConnected`.
  - `modes`: A tuple of modes used to construct `FourierFeature`.

# Keyword Arguments

  - `modes`: A tuple of modes used to construct `FourierFeature`.

# Examples

```julia
m = MultiscaleFourier(2, (30, 30, 1), sin; modes=(1 => 10, 10 => 10, 50 => 10))
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
    fc = FullyConnected(fourierfeature.out_dims, out_dims, activation)
    return Chain(fourierfeature, fc)
end

"""
    Siren(in_dims::Int, hidden_dim::Int, num_layers::Int; omega = 30f0)
    Siren(in_dims::Int, hidden_dims::NTuple{N, T}; omega = 30f0) where {N, T <: Int}

Sinusoidal Representation Network.

## Keyword Arguments

  - `omega`: The `ω₀` used for the first layer.

## References

[1] Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in Neural Information Processing Systems 33 (2020): 7462-7473.
"""
function Siren(in_dims::Int, hidden_dim::Int, num_layers::Int; omega=30.0f0)
    return Siren(in_dims, ntuple(i -> hidden_dim, num_layers); omega=omega)
end

@generated function Siren(in_dims::Int, hidden_dims::NTuple{N, T};
                          omega=30.0f0) where {N, T <: Int}
    N == 1 && return :(Sine(in_dims, hidden_dims[1]; is_first=true, omega=omega))
    get_layer(i) = :(Sine(hidden_dims[$i] => hidden_dims[$(i + 1)]))
    layers = [:(Sine(in_dims => hidden_dims[1]; is_first=true, omega=omega))]
    append!(layers, [get_layer(i) for i in 1:(N - 2)])
    append!(layers, [:(Sine(hidden_dims[$(N - 1)] => hidden_dims[$N], identity))])
    return :(Chain($(layers...)))
end
