"""
    PINNAttention(H_net, U_net, V_net, fusion=TriplewiseFusion)
    PINNAttention(H_net, U_net, V_net; fusion_layers)

The output dimesion of `H_net` and the input dimension of `layers` must be the same .

```
           x → U_net → u                     u
                         ↘                     ↘
x → H_net →  h1 → layer1 → connection → layer2 → connection
                         ↗                     ↗
           x → V_net → v                     v
```

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
                       V_net::AbstractExplicitLayer; fusion_layers::AbstractExplicitLayer)
    fusion = TriplewiseFusion(attention_connection, fusion_layers)
    return PINNAttention{typeof(H_net), typeof(U_net), typeof(V_net), typeof(fusion)}(H_net,
                                                                                      U_net,
                                                                                      V_net,
                                                                                      fusion)
end

function PINNAttention(int_dims::Int, out_dims::Int, activation::Function;
                       fusion_layers::AbstractExplicitLayer)
    activation = NNlib.fast_act(activation)
    H_net = Dense(int_dims, out_dims, activation)
    U_net = Dense(int_dims, out_dims, activation)
    V_net = Dense(int_dims, out_dims, activation)
    return PINNAttention(H_net, U_net, V_net; fusion_layers=fusion_layers)
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
x → [x; Fourier(x)] → PINNAttention(Dense, Dense, Dense, TriplewiseFusion)
```
"""

"""
    MultiscaleFourier(int_dims::Int, out_dims::NTuple, activation=identity, modes::NTuple)

Multi-scale Fourier Feature Net.

```
x → FourierFeature → FullyConnected → y
```

# Arguments

  - `int_dims`: The number of input dimensions.
  - `out_dims`: A tuple of output dimensions used to construct `FullyConnected`.
  - `activation`: The activation function used to construct `FullyConnected`.
  - `modes`: A tuple of modes used to construct `FourierFeature`.

# References

[1] Wang, Sifan, Hanwen Wang, and Paris Perdikaris. “On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks.” Computer Methods in Applied Mechanics and Engineering 384 (2021): 113938.
"""
function MultiscaleFourier(int_dims::Int,
                           out_dims::NTuple{N1, Int}=(ntuple(i -> 512, 6)..., 1),
                           activation::Function=identity,
                           modes::NTuple{N2, Pair{S, Int}}=(1 => 64, 10 => 64, 20 => 64,
                                                            50 => 32, 100 => 32)) where {N1,
                                                                                         N2,
                                                                                         S}
    fourierfeature = FourierFeature(int_dims, modes)
    chain = FullyConnected(fourierfeature.out_dims, out_dims, activation)
    return Chain(fourierfeature, chain)
end
