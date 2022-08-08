module Sophon

using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, AbstractExplicitLayer,
            AbstractExplicitContainerLayer

include("layers/basic.jl")
include("layers/nets.jl")

export FourierFeature, TriplewiseFusion, PINNAttentionNet, MultiscaleFourierNet
end
