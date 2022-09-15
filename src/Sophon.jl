module Sophon

using LinearAlgebra
using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, parameterlength, statelength,
            AbstractExplicitLayer, AbstractExplicitContainerLayer, zeros32

using Optimisers, Optimization, OptimizationOptimisers
import ParameterSchedulers: Step, Exp, Poly, Inv, Triangle, TriangleDecay2, TriangleExp,
                            Sin, SinDecay2, SinExp, CosAnneal, Sequence, Loop, Interpolator,
                            Shifted, ComposedSchedule, Constant
using ParameterSchedulers: AbstractSchedule
using SciMLBase, NeuralPDE, ComponentArrays
using StatsBase, QuasiMonteCarlo
using Adapt, ChainRulesCore, CUDA, GPUArrays, GPUArraysCore
using Setfield

include("layers/basic.jl")
include("layers/nets.jl")
include("utils.jl")
include("layers/containers.jl")
include("activations.jl")
include("training/scheduler.jl")
include("training/rad.jl")
include("training/causal.jl")
include("training/evo.jl")
include("compact/componentarrays.jl")
include("layers/operators.jl")

export GPUComponentArray64
export Scheduler
export gaussian, quadratic, laplacian, expsin, multiquadratic
export FourierFeature, TriplewiseFusion, FullyConnected, Sine, RBF, DiscreteFourierFeature
export PINNAttention, FourierNet, FourierAttention, Siren, FourierFilterNet, BACON
export DeepONet
export ChainState

end
