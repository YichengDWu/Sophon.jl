module Sophon

using LinearAlgebra
using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, parameterlength, statelength,
            AbstractExplicitLayer, AbstractExplicitContainerLayer, zeros32

import ModelingToolkit
using Optimization
import ParameterSchedulers: Step, Exp, Poly, Inv, Triangle, TriangleDecay2, TriangleExp,
                            Sin, SinDecay2, SinExp, CosAnneal, Sequence, Loop, Interpolator,
                            Shifted, ComposedSchedule, Constant
using ParameterSchedulers: AbstractSchedule
using ComponentArrays
import SciMLBase
import SciMLBase: parameterless_type, __solve, build_solution, NullParameters
using StatsBase, QuasiMonteCarlo
using Adapt, ChainRulesCore, CUDA, GPUArrays, GPUArraysCore
import QuasiMonteCarlo
import Sobol
using Memoize, LRUCache
using RuntimeGeneratedFunctions
using DomainSets, StaticArraysCore
import Symbolics
using ForwardDiff

RuntimeGeneratedFunctions.init(@__MODULE__)

include("layers/basic.jl")
include("layers/containers.jl")
include("layers/nets.jl")
include("utils.jl")
include("activations.jl")
include("compact/componentarrays.jl")
include("compact/NeuralPDE/pinn_types.jl")
include("compact/NeuralPDE/sym_utils.jl")
include("compact/NeuralPDE/utils.jl")
include("compact/NeuralPDE/training_strategies.jl")
include("compact/NeuralPDE/pinnsampler.jl")
include("compact/NeuralPDE/discretize.jl")
include("layers/operators.jl")

# Extensions
if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Optimisers="3bd65402-5787-11e9-1adc-39752487f4e2" begin
             include("../ext/OptimisersExt.jl")
         end
    end
end

export Scheduler
export gaussian, quadratic, laplacian, expsin, multiquadratic
export FourierFeature, TriplewiseFusion, FullyConnected, Sine, RBF, DiscreteFourierFeature,
       ConstantFunction, ScalarLayer, SplitFunction, FactorizedDense
export PINNAttention, FourierNet, FourierAttention, Siren, FourierFilterNet, BACON
export DeepONet
export PINN, symbolic_discretize, discretize, QuasiRandomSampler, NonAdaptiveTraining,
       AdaptiveTraining, ChainState

export get_global_ps, get_local_ps
end
