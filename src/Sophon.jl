module Sophon

using LinearAlgebra
using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, parameterlength, statelength,
            AbstractExplicitLayer, AbstractExplicitContainerLayer, zeros32

import ModelingToolkit
import ModelingToolkit: Differential
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
using Distributions: Beta
using Memoize, LRUCache
using RuntimeGeneratedFunctions
using DomainSets, StaticArraysCore
import Symbolics
using ForwardDiff
using MacroTools
using MacroTools: prewalk, postwalk
using Requires

RuntimeGeneratedFunctions.init(@__MODULE__)

include("layers/basic.jl")
include("layers/activations.jl")
include("layers/containers.jl")
include("layers/nets.jl")
include("layers/utils.jl")
include("layers/operators.jl")

include("pde/componentarrays.jl")
include("pde/pinn_types.jl")
include("pde/utils.jl")
include("pde/sym_utils.jl")
include("pde/training_strategies.jl")
include("pde/pinnsampler.jl")
include("pde/discretize.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Optimisers="3bd65402-5787-11e9-1adc-39752487f4e2" begin include("../ext/SophonOptimisersExt.jl") end
        @require TaylorDiff="b36ab563-344f-407b-a36a-4f200bebf99c" begin include("../ext/SophonTaylorDiffExt.jl") end
    end
end

export gaussian, quadratic, laplacian, expsin, multiquadratic, stan
export FourierFeature, TriplewiseFusion, FullyConnected, ResNet, Sine, RBF,
       DiscreteFourierFeature, ConstantFunction, ScalarLayer, SplitFunction, FactorizedDense
export PINNAttention, FourierNet, FourierAttention, Siren, FourierFilterNet, BACON
export DeepONet
export PINN, symbolic_discretize, discretize, QuasiRandomSampler, NonAdaptiveTraining,
       AdaptiveTraining, ChainState, BetaRandomSampler

export get_global_ps, get_local_ps
end
