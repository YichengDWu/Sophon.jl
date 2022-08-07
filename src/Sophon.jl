module Sophon

using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, AbstractExplicitLayer, AbstractExplicitContainerLayer

include("layers/basic.jl")

export Fourier
end
