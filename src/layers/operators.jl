"""
    DeepONet(branch_net, trunk_net;
             flatten_layer=FlattenLayer(),
             linear_layer=NoOpLayer(),
             bias=Scalar())
    DeepONet(layer_sizes_branch, activation_branch,
             layer_sizes_trunk,
             activation_trunk)

Deep operator network. Note that the branch net supports multi-dimensional inputs. The `flatten_layer`
flatten the output of the branch net to a matrix, and the `linear_layer` is applied to the flattened.
You would at least need to specify the `linear_layer` to transform the flattened matrix to the correct shape.

```
v → branch_net → flatten_layer → linear_layer → b
                                                  ↘
                                                    b' * t + bias -> u
                                                  ↗
                                ξ → trunk_net → t
```

## Arguments

  - `branch_net`: The branch net.
  - `trunk_net`: The trunk net.

## Keyword Arguments

  - `flatten_layer`: The layer to flatten a multi-dimensional array to a matrix.
  - `linear_layer`: The layer to apply a linear transformation to the output of the `flatten_layer`.

## Inputs

  - `(v, ξ)`: `v` is an array of shape ``(b_1,b_2,...b_d, m)``, where `d` is the dimension
    of the input function, and `m` is the number of input functions. `ξ` is a matrix of shape ``(d', n)``,
    where `d'` is the dimension of the output function, and `m` is the number of "sensors".

## Returns

  - A matrix of shape ``(m, n)``.

## Examples

```julia
julia> deeponet = DeepONet((3, 5, 4), relu, (2, 6, 4, 4), tanh)
DeepONet(
    branch_net = Chain(
        layer_1 = Dense(3 => 5, relu),  # 20 parameters
        layer_2 = Dense(5 => 4),        # 24 parameters
    ),
    trunk_net = Chain(
        layer_1 = Dense(2 => 6, tanh_fast),  # 18 parameters
        layer_2 = Dense(6 => 4, tanh_fast),  # 28 parameters
        layer_3 = Dense(4 => 4, tanh_fast),  # 20 parameters
    ),
    flatten_layer = FlattenLayer(),
    linear_layer = NoOpLayer(),
    bias = Scalar(),                    # 1 parameters
)         # Total: 111 parameters,
          #        plus 0 states, summarysize 80 bytes.
```

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

function DeepONet(layer_sizes_branch::NTuple{N1, T}, activation_branch::Function,
                  layer_sizes_trunk::NTuple{N2, T},
                  activation_trunk::Function) where {T <: Int, N1, N2}
    @assert last(layer_sizes_branch)==last(layer_sizes_trunk) "Output sizes of branch net and trunk net must match"
    branch_net = FullyConnected(layer_sizes_branch, activation_branch)
    trunk_net = FullyConnected(layer_sizes_trunk, activation_trunk; outermost=false)

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
