"""
    PINNAttention(H_net, U_net, V_net, fusion::TriplewiseFusion)
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
struct PINNAttention <: AbstractExplicitContainerLayer{(:H_net, :U_net, :V_net, :fusion)}
    H_net::AbstractExplicitLayer
    U_net::AbstractExplicitLayer
    V_net::AbstractExplicitLayer
    fusion::AbstractExplicitLayer
    function PINNAttention(H_net::AbstractExplicitLayer, U_net::AbstractExplicitLayer,
                           V_net::AbstractExplicitLayer, fusion::TriplewiseFusion)
        return new(H_net, U_net, V_net, fusion)
    end
end

function PINNAttention(H_net::AbstractExplicitLayer, U_net::AbstractExplicitLayer,
                       V_net::AbstractExplicitLayer; fusion_layers::AbstractExplicitLayer)
    fusion = TriplewiseFusion(attention_connection, fusion_layers)
    return PINNAttention(H_net, U_net, V_net, fusion)
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
    MultiscaleFourier(int_dims::Int, out_dims::Int, hidden_dims::Int, chain; std)

Multi-scale Fourier Net.

```
    FourierFeature → chain_1
  ↗                          ↘
x → FourierFeature → chain_2 → Dense → y
  ↘                          ↗
    FourierFeature → chain_3
```

Here `chain_1`, `chain_2`, and `chain_3` are all identical copies of `chain`, but with different parameters.

# Keyword Arguments

  - `std`: A container of standard deviations of the Gaussian distribution.

# References

[1] Wang, Sifan, Hanwen Wang, and Paris Perdikaris. “On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks.” Computer Methods in Applied Mechanics and Engineering 384 (2021): 113938.
"""
struct MultiscaleFourier{T, C <: AbstractExplicitLayer, O} <:
       AbstractExplicitContainerLayer{(:chains, :output_layer)}
    std::T
    chains::C
    output_layer::O
    int_dims::Int
    out_dims::Int
    hidden_dims::Int
end

function MultiscaleFourier(int_dims::Int, out_dims::Int, hidden_dims::Int, chain; std)
    M = length(std)
    chains = [Chain(FourierFeature(int_dims, hidden_dims; std=i), chain) for i in std]
    chains = BranchLayer(chains...)
    output_layer = Dense(hidden_dims * M, out_dims)
    return MultiscaleFourier{typeof(std), typeof(chains), typeof(output_layer)}(std, chains,
                                                                                output_layer,
                                                                                int_dims,
                                                                                out_dims,
                                                                                hidden_dims)
end

function (m::MultiscaleFourier)(x::AbstractArray, ps, st::NamedTuple)
    hs, st_chains = m.chains(x, ps.chains, st.chains)
    y, st_out = m.output_layer(vcat(hs...), ps.output_layer, st.output_layer)
    st = merge(st, (chains=st_chains, output_layer=st_out))
    return y, st
end
