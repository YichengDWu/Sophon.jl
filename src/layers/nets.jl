"""
    PINNAttention(H_net, U_net, V_net, fusion_layers)
    PINNAttention(in_dim, hidden_dims, activation, fusion_layers)

The output dimesion of `H_net` and the input dimension of `fusion_layers` must be the same.
For the second constructor, `Dense` layers is used for `H_net`, `U_net`, and `V_net`.

```
                 x → U_net → u                           u
                               ↘                           ↘
x → H_net →  h1 → fusionlayer1 → connection → fusionlayer2 → connection
                               ↗                           ↗
                 x → V_net → v                           v
```

# Arguments

    - `H_net`: `AbstractExplicitLayer`
    - `U_net`: `AbstractExplicitLayer`
    - `V_net`: `AbstractExplicitLayer`
    - `in_dim`: The input dimension.
    - `hidden_dims`: The output dimension of `H_net`.
    - `fusion_layers`: `AbstractExplicitLayer` or a tuple of integeters. In the latter case,
    fully connected layers are used.

# References

[1] Wang, Sifan, Yujun Teng, and Paris Perdikaris. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081
"""
struct PINNAttention{H, U, V, F} <:
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

function PINNAttention(in_dim::Int, hidden_dims::Int, activation::Function,
                       fusion_layers::T) where {T}
    H_net = Dense(in_dim, hidden_dims, activation)
    U_net = Dense(in_dim, hidden_dims, activation)
    V_net = Dense(in_dim, hidden_dims, activation)
    fusion_layers = T <: NTuple ? FullyConnected(hidden_dims, fusion_layers, activation) :
                    fusion_layers
    return PINNAttention(H_net, U_net, V_net, fusion_layers)
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
    FourierAttention()

```
x → [x; FourierFeature(x)] → PINNAttention(Dense, Dense, Dense, TriplewiseFusion)
```
"""
function FourierAttention()
    return input_layer = PairwiseFusion
end

"""
    MultiscaleFourier(in_dim::Int, out_dim::NTuple, activation=identity, modes::NTuple)

Multi-scale Fourier Feature Net.

```
x → FourierFeature → FullyConnected → y
```

# Arguments

  - `in_dim`: The number of input dimensions.
  - `out_dim`: A tuple of output dimensions used to construct `FullyConnected`.
  - `activation`: The activation function used to construct `FullyConnected`.
  - `modes`: A tuple of modes used to construct `FourierFeature`.

# References

[1] Wang, Sifan, Hanwen Wang, and Paris Perdikaris. “On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks.” Computer Methods in Applied Mechanics and Engineering 384 (2021): 113938.
"""
function MultiscaleFourier(in_dim::Int,
                           out_dim::NTuple{N1, Int}=(ntuple(i -> 512, 6)..., 1),
                           activation::Function=identity,
                           modes::NTuple{N2, Pair{S, Int}}=(1 => 64, 10 => 64, 20 => 64,
                                                            50 => 32, 100 => 32)) where {N1,
                                                                                         N2,
                                                                                         S}
    fourierfeature = FourierFeature(in_dim, modes)
    chain = FullyConnected(fourierfeature.out_dim, out_dim, activation)
    return Chain(fourierfeature, chain)
end
