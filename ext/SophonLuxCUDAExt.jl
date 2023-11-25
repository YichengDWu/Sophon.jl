module SophonLuxCUDAExt

using Lux, LuxCUDA, Sophon, Optimization, Adapt

function (::LuxCUDADevice)(prob::OptimizationProblem)
    u0 = adapt(CuArray, prob.u0)
    p = Tuple(adapt(CuArray, prob.p[i]) for i in 1:length(prob.p))  # have to use tuple here...
    return Optimization.OptimizationProblem(prob.f, u0, p)
end

end
