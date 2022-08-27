"""
    DeepONet(branch_net, trunk_net;
             flatten_layer=FlattenLayer(),
             linear_layer=NoOpLayer(),
             bias=Scalar())

Deep operator network.

## Reference

[lu2019deeponet](@cite)
"""
struct DeepONet{B, T, F, L, S} <:
       AbstractExplicitContainerLayer{(:branch_net, :trunk_net, :flatten_layer,
                                       :linear_layer, :bias)}
    branch_net::B
    trunk_net::T
    flatten_layer::F
    linear_layer::L
    bias::S
end

function DeepONet(branch_net::AbstractExplicitLayer, trunk_net::AbstractExplicitLayer;
                  flatten_layer::AbstractExplicitLayer=FlattenLayer(),
                  linear_layer::AbstractExplicitLayer=NoOpLayer(),
                  bias::AbstractExplicitLayer=Scalar())
    return DeepONet{typeof(branch_net), typeof(trunk_net), typeof(flatten_layer),
                    typeof(linear_layer), typeof(bias)}(branch_net, trunk_net,
                                                        flatten_layer, linear_layer, bias)
end

function DeepONet(layer_dims_branch::NTuple{N1, T}, activation_branch::Function,
                  layer_dims_trunk::NTuple{N2, T},
                  activation_trunk::Function) where {T <: Int, N1, N2}
    @assert last(layer_dims_branch)==last(layer_dims_trunk) "Output sizes of branch net and trunk net must match"
    branch_net = FullyConnected(layer_dims_branch, activation_branch)
    trunk_net = FullyConnected(layer_dims_trunk, activation_trunk; outermost=false)

    return DeepONet(branch_net, trunk_net)
end

function (m::DeepONet)(x::Tuple{AbstractArray, AbstractVecOrMat}, ps, st::NamedTuple)
    b, st_branch_net = m.branch_net(first(x), ps.branch_net, st.branch_net)
    t, st_trunk_net = m.trunk_net(last(x), ps.trunk_net, st.trunk_net)

    b, st_flatten_layer = m.flatten_layer(b, ps.flatten_layer, st.flatten_layer)
    b, st_linear_layer = m.linear_layer(b, ps.linear_layer, st.linear_layer)
    st = merge(st,
               (branch_net=st_branch_net, trunk_net=st_trunk_net,
                linear_layer=st_linear_layer, flatten_layer=st_flatten_layer))
    return transpose(b) * t .+ ps.bias.scalar, st
end

function (m::DeepONet{B, T, F, NoOpLayer})(x::Tuple{AbstractArray, AbstractVecOrMat}, ps,
                                           st::NamedTuple) where {B, T, F}
    b, st_branch_net = m.branch_net(first(x), ps.branch_net, st.branch_net)
    t, st_trunk_net = m.trunk_net(last(x), ps.trunk_net, st.trunk_net)

    st = merge(st, (branch_net=st_branch_net, trunk_net=st_trunk_net))
    return transpose(b) * t .+ ps.bias.scalar, st
end
