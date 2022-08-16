Base.getindex(c::Chain, i::Int) = c.layers[i]
Base.length(c::Chain) = length(c.layers)

@inline GPUComponentArray64(nt::NamedTuple) = nt |> ComponentArray |> gpu .|> Float64
