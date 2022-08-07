"""
    PINNAttentionNet(H_net, U_net, V_net, fusion::TriplewiseFusion)
    PINNAttentionNet(H_net, U_net, V_net; fusion_layers)

The output dimesion of `H_net` and the input dimension of `layers` must be the same .

# References

[1] Wang, Sifan, Yujun Teng, and Paris Perdikaris. "Understanding and mitigating gradient flow pathologies in physics-informed neural networks." SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081
"""
struct PINNAttentionNet <: AbstractExplicitContainerLayer{(:H_net, :U_net, :V_net, :fusion)}
    H_net::AbstractExplicitLayer
    U_net::AbstractExplicitLayer
    V_net::AbstractExplicitLayer
    fusion::AbstractExplicitLayer
    function PINNAttentionNet(H_net::AbstractExplicitLayer, U_net::AbstractExplicitLayer,
                              V_net::AbstractExplicitLayer, fusion::TriplewiseFusion)
        return new(H_net, U_net, V_net, fusion)
    end
end

function PINNAttentionNet(H_net::AbstractExplicitLayer, U_net::AbstractExplicitLayer,
                          V_net::AbstractExplicitLayer;
                          fusion_layers::AbstractExplicitLayer)
    fusion = TriplewiseFusion(attention_connection, fusion_layers)
    return PINNAttentionNet(H_net, U_net, V_net, fusion)
end

function PINNAttentionNet(int_dims::Int, out_dims::Int, activation::Function;
                          fusion_layers::AbstractExplicitLayer)
    activation = NNlib.fast_act(activation)
    H_net = Dense(int_dims, out_dims, activation)
    U_net = Dense(int_dims, out_dims, activation)
    V_net = Dense(int_dims, out_dims, activation)
    return PINNAttentionNet(H_net, U_net, V_net; fusion_layers=fusion_layers)
end

function (m::PINNAttentionNet)(x::AbstractArray, ps, st::NamedTuple)
    H, st_h = m.H_net(x, ps.H_net, st.H_net)
    U, st_u = m.U_net(x, ps.U_net, st.U_net)
    V, st_v = m.V_net(x, ps.V_net, st.V_net)
    F, st_f = m.fusion((H, U, V), ps.fusion, st.fusion)
    st = merge(st, (U_net=st_u, V_net=st_v, H_net=st_h, fusion=st_f))
    return F, st
end

attention_connection(z, u, v) = (1 .- z) .* u .+ z .* v
