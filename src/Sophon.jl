module Sophon

using LinearAlgebra
using Lux, Random, NNlib, NNlibCUDA, GPUArrays
import Lux: initialparameters, initialstates, AbstractExplicitLayer,
            AbstractExplicitContainerLayer

using Optimisers, Optimization, OptimizationOptimisers
import ParameterSchedulers: Step, Exp, Poly, Inv, Triangle, TriangleDecay2, TriangleExp,
                            Sin, SinDecay2, SinExp, CosAnneal, Sequence, Loop, Interpolator,
                            Shifted, ComposedSchedule

using SciMLBase, NeuralPDE, ComponentArrays
using StatsBase, QuasiMonteCarlo
using Adapt, ChainRulesCore

include("layers/basic.jl")
include("layers/nets.jl")
include("utils.jl")
include("activations.jl")
include("scheduler.jl")
include("training/rad.jl")
include("training/causal.jl")
include("compact/componentarrays.jl")

export GPUComponentArray64
export Scheduler, get_opt
export RADTraining
#export gaussian, quadratic, laplacian, expsin
export RADTraining, CausalTraining
export FourierFeature, TriplewiseFusion, FullyConnected, Sine, RBF
export PINNAttention, MultiscaleFourier, FourierAttention, Siren, SirenAttention

end
