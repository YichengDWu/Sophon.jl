module Sophon

using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, AbstractExplicitLayer,
            AbstractExplicitContainerLayer

include("layers/basic.jl")
include("layers/nets.jl")
include("utils.jl")

export FourierFeature, TriplewiseFusion, FullyConnected
export PINNAttention, MultiscaleFourier, FourierAttention
end
