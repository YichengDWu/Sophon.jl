module Sophon

using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, AbstractExplicitLayer,
            AbstractExplicitContainerLayer

using Optimisers, Optimization, OptimizationOptimisers
import ParameterSchedulers: Step, Exp, Poly, Inv, Triangle, TriangleDecay2, TriangleExp, Sin,
                           SinDecay2, SinExp, CosAnneal, Sequence, Loop, Interpolator,
                           Shifted, ComposedSchedule

using SciMLBase, NeuralPDE, ComponentArrays
using StatsBase, QuasiMonteCarlo
using Adapt

include("layers/basic.jl")
include("layers/nets.jl")
include("utils.jl")
include("activations.jl")
include("scheduler.jl")
include("training/rad.jl")


export Scheduler, get_opt
#export gaussian, quadratic, laplacian, expsin
export RADTraining
export FourierFeature, TriplewiseFusion, FullyConnected, Sine
export PINNAttention, MultiscaleFourier, FourierAttention, Siren, SirenAttention
end
