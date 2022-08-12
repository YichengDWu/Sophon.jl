module Sophon

using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, AbstractExplicitLayer,
            AbstractExplicitContainerLayer

include("layers/basic.jl")
include("layers/nets.jl")
include("utils.jl")
include("activations.jl")

#export gaussian, quadratic, laplacian, expsin
export FourierFeature, TriplewiseFusion, FullyConnected, Sine
export PINNAttention, MultiscaleFourier, FourierAttention, Siren
end
