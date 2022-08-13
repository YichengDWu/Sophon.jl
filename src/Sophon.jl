module Sophon

using Lux, Random, NNlib, NNlibCUDA
import Lux: initialparameters, initialstates, AbstractExplicitLayer,
            AbstractExplicitContainerLayer
using Optimisers
import ParameterSchedulers: Step, Exp, Poly, Inv, Triangle, TriangleDecay2, TriangleExp, Sin,
                           SinDecay2, SinExp, CosAnneal, Sequence, Loop, Interpolator,
                           Shifted, ComposedSchedule

using SciMLBase

include("layers/basic.jl")
include("layers/nets.jl")
include("utils.jl")
include("activations.jl")
include("scheduler.jl")

export Scheduler, get_opt
#export gaussian, quadratic, laplacian, expsin
export FourierFeature, TriplewiseFusion, FullyConnected, Sine
export PINNAttention, MultiscaleFourier, FourierAttention, Siren, SirenAttention
end
